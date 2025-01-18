#!/usr/bin/env python

import tensorflow._api.v2.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
import sys
import numpy as np

# Don't generate pyc codes
sys.dont_write_bytecode = True

def concatenation(nodes):
    return tf.concat(nodes, axis=3)

def denseBlock(Img, num_layers, len_dense, num_filters, kernel_size, downsampling=False):
    with tf.variable_scope("dense_unit" + str(num_layers), reuse=tf.AUTO_REUSE):
        nodes = []
        img = tf.layers.conv2d(inputs=Img, padding='same', filters=num_filters, kernel_size=kernel_size, activation=None)
        nodes.append(img)
        for z in range(len_dense):
            img = tf.nn.relu(Img)
            img = tf.layers.conv2d(inputs=img, padding='same', filters=num_filters, kernel_size=kernel_size, activation=None)
            net = tf.layers.conv2d(inputs=concatenation(nodes), padding='same', filters=num_filters, kernel_size=kernel_size, activation=None)
            nodes.append(net)
        return net

def CIFAR10Model(Img, ImageSize, MiniBatchSize):
    """
    Inputs:
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """
    # Define Filter parameters for the first conv layer block
    filter_size1 = 5
    num_filters1 = 16
    n1 = 1  # number of residual blocks in each con layer blocks

    # Define Filter parameters for the second conv layer block
    filter_size2 = 3
    num_filters2 = 16
    n2 = 2  # number of residual blocks in each conv layer blocks

    # Define number of class labels
    num_classes = 10

    # Define net placeholder
    net = Img
    net = tf.layers.conv2d(net, num_filters1, kernel_size=filter_size1, activation=None, padding='same')

    # Construct first convolution block of n residual blocks
    net = denseBlock(Img=net, num_layers=1, len_dense=4, num_filters=num_filters1, kernel_size=filter_size1, downsampling=False)
    net = tf.layers.conv2d(net, num_filters1, kernel_size=filter_size1, activation=None, padding='same')
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    net = denseBlock(Img=net, num_layers=2, len_dense=4, num_filters=num_filters2, kernel_size=filter_size2, downsampling=False)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    # Define flatten_layer
    net = tf.layers.flatten(net)

    # Define the Neural Network's fully connected layers:
    net = tf.layers.dense(inputs=net, units=128, activation=tf.nn.relu, name='fc1_layer')
    net = tf.layers.dense(inputs=net, units=256, activation=tf.nn.relu, name='fc2_layer')
    net = tf.layers.dense(inputs=net, units=num_classes, activation=None, name='fc_out_layer')

    # prLogits is defined as the final output of the neural network
    prLogits = net
    # prSoftMax is defined as normalized probabilities of the output of the neural network
    prSoftMax = tf.nn.softmax(logits=prLogits)

    return prLogits, prSoftMax