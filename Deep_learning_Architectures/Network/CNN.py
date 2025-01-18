#!/usr/bin/env python

import tensorflow._api.v2.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
import sys
import numpy as np

# Don't generate pyc codes
sys.dont_write_bytecode = True

def CIFAR10Model(Img, ImageSize, MiniBatchSize):
  
    # Define number of class labels
    num_classes = 10

    # Construct first convolution layer
    net = Img
    net = tf.layers.conv2d(inputs=net, name='conv1_layer', padding='same', filters=32, kernel_size=5, activation=tf.nn.relu)
    net = tf.layers.conv2d(inputs=net, name='conv2_layer', padding='same', filters=32, kernel_size=5, activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    net = tf.layers.conv2d(inputs=net, name='conv3_layer', padding='same', filters=64, kernel_size=5, activation=tf.nn.relu)
    net = tf.layers.conv2d(inputs=net, name='conv4_layer', padding='same', filters=32, kernel_size=5, activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    net = tf.layers.flatten(net)

    # Define the Neural Network's fully connected layers:
    net = tf.layers.dense(inputs=net, name='fc1_layer', units=128, activation=tf.nn.relu)
    net = tf.layers.dense(inputs=net, name='fc2_layer', units=256, activation=tf.nn.relu)
    net = tf.layers.dense(inputs=net, name='fc_out_layer', units=num_classes, activation=None)

    # prLogits is defined as the final output of the neural network
    prLogits = net
    # prSoftMax is defined as normalized probabilities of the output of the neural network
    prSoftMax = tf.nn.softmax(logits=prLogits)

    return prLogits, prSoftMax
