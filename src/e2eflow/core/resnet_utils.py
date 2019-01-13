# implementation of residual units from
# https://github.com/tensorflow/models/blob/master/research/struct2depth/nets.py

import tensorflow as tf
import numpy as np

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'


def _conv(x, filter_size, out_channel, stride, pad='SAME', name='conv'):
    """Helper function for defining ResNet architecture."""
    in_shape = x.get_shape()
    with tf.variable_scope(name):
        # Main operation: conv2d
        with tf.device('/CPU:0'):
            kernel = tf.get_variable(
                'kernel', [filter_size, filter_size, in_shape[3], out_channel],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / filter_size / filter_size / out_channel)))
        if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
        conv = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], pad)
    return conv


def _bn(x, is_train, name='bn'):
    """Helper function for defining ResNet architecture."""
    bn = tf.layers.batch_normalization(x, training=is_train, name=name)
    return bn


def _relu(x, name=None, leakness=0.0):
    """Helper function for defining ResNet architecture."""
    if leakness > 0.0:
        name = 'lrelu' if name is None else name
        return tf.maximum(x, x * leakness, name='lrelu')
    else:
        name = 'relu' if name is None else name
        return tf.nn.relu(x, name='relu')


def _residual_block(x, is_training, name='unit'):
    """Helper function for defining ResNet architecture."""
    num_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        shortcut = x  # Shortcut connection
        # Residual
        x = _conv(x, 3, num_channel, 1, name='conv_1')
        x = _bn(x, is_train=is_training, name='bn_1')
        x = _relu(x, name='relu_1')
        x = _conv(x, 3, num_channel, 1, name='conv_2')
        x = _bn(x, is_train=is_training, name='bn_2')
        # Merge
        x = x + shortcut
        x = _relu(x, name='relu_2')
        return x


def _residual_block_encode(x, is_training, out_channel, strides, name='unit'):
    """Helper function for defining ResNet architecture."""
    in_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        # Shortcut connection
        if in_channel == out_channel:
            if strides == 1:
                shortcut = tf.identity(x)
            else:
                shortcut = tf.nn.max_pool(x, [1, strides, strides, 1],
                                          [1, strides, strides, 1], 'VALID')
        else:
            shortcut = _conv(x, 1, out_channel, strides, name='shortcut')
        # Residual
        x = _conv(x, 3, out_channel, strides, name='conv_1')
        x = _bn(x, is_train=is_training, name='bn_1')
        x = _relu(x, name='relu_1')
        x = _conv(x, 3, out_channel, 1, name='conv_2')
        x = _bn(x, is_train=is_training, name='bn_2')
        # Merge
        x = x + shortcut
        x = _relu(x, name='relu_2')
        return x
