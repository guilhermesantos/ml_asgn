#!/usr/bin/env python
# coding: utf-8

# Deep Learning
# =============
# 
# Assignment 4
# ------------
# 
# Previously in `2_fullyconnected.ipynb` and `3_regularization.ipynb`, we trained fully connected networks to classify [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) characters.
# 
# The goal of this assignment is make the neural network convolutional.

# In[1]:


# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import os


# In[2]:


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


# Reformat into a TensorFlow-friendly shape:
# - convolutions need the image data formatted as a cube (width by height by #channels)
# - labels as float 1-hot encodings.

# In[3]:


image_size = 28
num_labels = 10
num_channels = 1 # grayscale

import numpy as np

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# In[4]:


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# Let's build a small network with two convolutional layers, followed by one fully connected layer. Convolutional networks are more expensive computationally, so we'll limit its depth and number of fully connected nodes.

# In[ ]:


batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  # Model.
  def model(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases
  
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))


# In[ ]:


num_steps = 1001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


# ---
# Problem 1
# ---------
# 
# The convolutional model above uses convolutions with stride 2 to reduce the dimensionality. Replace the strides by a max pooling operation (`nn.max_pool()`) of stride 2 and kernel size 2.
# 
# ---

# In[ ]:


num_steps = 5001
print_period = 1000
batch_size = 16
num_channels = 1
patch_size = 5
depth = 16
num_hidden = 64
beta = 0.5
train_subset = 200000
init_train_rate = 0.5

graph = tf.Graph()
with graph.as_default():
    #Loading input data into the model
    train_x = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, num_channels])
    train_y = tf.placeholder(tf.float32, shape=[batch_size, num_labels])
    
    valid_x = tf.constant(valid_dataset)
    test_x = tf.constant(test_dataset)
    
    #Setting layers up
    w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    b1 = tf.Variable(tf.zeros([depth]))
    w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0, shape=[depth]))
    w3 = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    b3 = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    w4 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    b4 = tf.Variable(tf.constant(1.0, shape=[num_labels]))
    
    #Logits
    def predict(data):
        conv_strides = [1, 1, 1, 1]
        pool_strides =  [1, 2, 2, 1]
        #First convolutional layer
        l1 = tf.nn.conv2d(data, w1, conv_strides, padding='SAME')
        o1 = tf.nn.relu(l1+b1)
        p1 = tf.nn.max_pool(o1, pool_strides, pool_strides, padding='SAME')
        
        #Second convolutional layer
        l2 = tf.nn.conv2d(p1, w2, conv_strides, padding='SAME')
        o2 = tf.nn.relu(l2+b2)
        p2 = tf.nn.max_pool(o2, pool_strides, pool_strides, padding='SAME')
        
        #Fully connected layer
        p2_shape = p2.get_shape().as_list()
        print(type(p2.get_shape()))
        reshaped_p2 = tf.reshape(p2, [p2_shape[0], p2_shape[1]*p2_shape[2]*p2_shape[3]])
        l3 = tf.matmul(reshaped_p2, w3)+b3
        o3 = tf.nn.relu(l3)
        l4 = tf.matmul(o3, w4)+b4
        return l4
    
    logits = predict(train_x)
    
    #Loss function calculation and minimization
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_y, logits=logits))
    loss_reg3 = tf.nn.l2_loss(w3)
    loss_reg4 = tf.nn.l2_loss(w4)
    loss = loss + beta*(loss_reg3 + loss_reg4)
    
    #Training rate update
    #global_step = tf.Variable(0)
    #irt = init_train_rate
    #learning_rate =  tf.train.exponential_decay(irt, global_step, 100000, 0.96, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    
    train_py = tf.nn.softmax(logits)
    valid_py = tf.nn.softmax(predict(valid_x))
    test_py = tf.nn.softmax(predict(test_x))


# In[ ]:


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    
    for step in range(num_steps):
        indexes = np.random.randint(low=0, high=train_subset, size=batch_size)

        batch_data = train_dataset[indexes]
        print('id do batch', id(batch_data))
        batch_labels = train_labels[indexes]
        
        f_dict = {train_x: batch_data, train_y : batch_labels}
        _, l, train_pred = session.run([optimizer, loss, train_py], feed_dict = f_dict)
        
        if(step % print_period == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(train_pred, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_py.eval(), valid_labels))
            print("Testing accuracy: %.1f%%" % accuracy(test_py.eval(), test_labels))
            print('\n')


# ---
# Problem 2
# ---------
# 
# Try to get the best performance you can using a convolutional net. Look for example at the classic [LeNet5](http://yann.lecun.com/exdb/lenet/) architecture, adding Dropout, and/or adding learning rate decay.
# 
# ---

# In[ ]:




