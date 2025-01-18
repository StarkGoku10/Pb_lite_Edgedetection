#!/usr/bin/env python

import tensorflow._api.v2.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
import sys
import numpy as np

# Don't generate pyc codes
sys.dont_write_bytecode = True

#Define Residual Network Block
def name_block(s,b):
    """
    s is the integer for the number of each set of residual blocks
    """
    s = 'block'+str(b)+'layer'+str(s)
    return s

def res_block(Img, num_filters, kernel_size,s,b, downsampling = True):
    """
    net is a MiniBatch of the current image
    num_filters is number of filters used in this layer
    kernel_size is the size of filter used
    s is the integer number of the block
    b is the integer number of the set of blocks with same parameters
    """
    name = name_block(s=s,b=b)

    im_store = tf.layers.conv2d(inputs = Img, name=name+'conv_', padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)
    im_store = tf.layers.batch_normalization(inputs = im_store,axis = -1, center = True, scale = True, name = name+'bn_')
    I_store = im_store

    #Define 1st layer of convolution
    net = tf.layers.conv2d(inputs = Img, name=name +'conv_1_', padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name =name +'bn1_')
    net = tf.nn.relu(net, name = name +'Relu1_layer')

    #Define 2nd Layer of the convolution
    net = tf.layers.conv2d(inputs = net, name=name+'conv_2_', padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name =name +'bn2_')

    if downsampling:
        net  = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2)
        I_store = tf.layers.max_pooling2d(inputs = I_store, pool_size = 2, strides = 2)
        print('\n')
        print('max_pool is on')
    out = tf.math.add(net, I_store)

    net = tf.nn.relu(out, name = name +'Relu2_')
    return net

def n_res_block(net, num_filters, kernel_size, n_blocks, b, downsampling =False):
    """
    net is a MiniBatch of the current image
    num_filters is number of filters used in this layer
    kernel_size is the size of filter used
    n_blocks is number of residual blocks with same num_filters and kernel_size
    b is the integer number of the set of blocks being used
    """
    for i in range(n_blocks):
        net = res_block(Img = net, num_filters = num_filters, kernel_size = kernel_size,s = i,b = b, downsampling = downsampling )
        net = net
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
    #Define Filter parameters for the first conv layer block
    filter_size1 = 3
    num_filters1 = 10
    n1 = 2                  #number of residual blocks in each conv layer blocks

    #Define Filter parameters for the second conv layer block
    filter_size2 = 3
    num_filters2 = 20
    n2 = 3                 #number of residual blocks in each conv layer blocks

    #Define Filter parameters for the second conv layer block
    filter_size3 = 4
    num_filters3 = 10
    n3 = 2                  #number of residual blocks in each conv layer blocks

    #Define number of class labels
    num_classes = 10

    #Define net placeholder
    net = Img
    #Construct first convolution block of n residual blocks
    net = n_res_block(net, num_filters = num_filters1, kernel_size = filter_size1, n_blocks = n1, b=1, downsampling =False)
    net = n_res_block(net, num_filters = num_filters2, kernel_size = filter_size2, n_blocks = n2, b=2, downsampling =False)
    net = n_res_block(net, num_filters = num_filters3, kernel_size = filter_size3, n_blocks = n3, b=3, downsampling =False)

    #Define flatten_layer
    net = tf.layers.flatten(net)
    net = tf.layers.dense(inputs = net, name ='layer_fc1', units = 256, activation = tf.nn.relu)
    net = tf.layers.dense(inputs = net, name ='layer_fc2',units=128, activation=tf.nn.relu)
    #net = tf.layers.dense(inputs = net, name ='layer_fc3',units=256, activation=tf.nn.relu)
    net = tf.layers.dense(inputs = net, name='layer_fc_out', units = num_classes, activation = None)

    #prLogits is defined as the final output of the neural network
    prLogits = net
    #prSoftMax is defined as normalized probabilities of the output of the neural network
    prSoftMax = tf.nn.softmax(logits = prLogits)

    return prLogits, prSoftMax
