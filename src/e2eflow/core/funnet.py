import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

from .alexnet import alexnet_v2, alexnet_v2_arg_scope
from .util import add_to_summary
from .util import epipolar_errors
from .util import get_fundamental_matrix

slim = tf.contrib.slim


def _leaky_relu(x):
    with tf.variable_scope('leaky_relu'):
        return tf.maximum(0.1 * x, x)


# def funnet(flow, trainable=False):
#     """
#
#     :param flow: input tensor with shape(batch_size, height, width, 2)
#     :param trainable: True if should be trained
#     :return: motion prediction of shape(batch_size, 5) with angles roll, pitch, yaw, trans_yaw, trans_pitch
#             all in range(-pi/2, pi/2)
#     """
#     # _, height, width, _ = flow.shape.as_list()
#     with tf.variable_scope('FunNet'):
#         # concat_inputs = tf.concat([inputs['input_a'],
#         #                           inputs['input_b'],
#         #                           inputs['brightness_error_sd'],
#         #                           inputs['brightness_error_css'],
#         #                           inputs['flow']], axis=3)
#         with slim.arg_scope([slim.conv2d],
#                             # Only backprop this network if trainable
#                             trainable=trainable,
#                             data_format='NCHW',
#                             weights_regularizer=slim.l2_regularizer(0.0004),
#                             weights_initializer=layers.variance_scaling_initializer(),
#                             biases_initializer=init_ops.constant_initializer(0.1),
#                             activation_fn=_leaky_relu
#                             ):
#             with slim.arg_scope(
#                     [slim.conv2d, layers.max_pool2d]):
#                 # with slim.arg_scope([slim.conv2d], stride=2):
#                 conv_1 = slim.conv2d(flow, 64, [11, 11], 4, scope='conv1')
#                 conv_1 = layers.max_pool2d(conv_1, [3, 3], 2, scope='pool1')
#                 conv_2 = slim.conv2d(conv_1, 192, [5, 5], scope='conv2')
#                 conv_2 = layers.max_pool2d(conv_2, [3, 3], 2, scope='pool2')
#                 conv_3 = slim.conv2d(conv_2, 384, [3, 3], scope='conv3')
#                 conv_4 = slim.conv2d(conv_3, 384, [3, 3], scope='conv4')
#                 conv_5 = slim.conv2d(conv_4, 256, [3, 3], scope='conv5')
#                 conv_5 = layers.max_pool2d(conv_5, [3, 3], 2, scope='pool5')
#
#                 print(conv_5.shape.as_list())
#
#             # Use conv2d instead of fully_connected layers.
#             dropout_keep_prob = 0.5
#             with slim.arg_scope(
#                     [slim.conv2d],
#                     weights_initializer=init_ops.truncated_normal_initializer(0.0, 0.005)):
#                 net = slim.conv2d(conv_5, 4096, [5, 5], scope='fc6')
#                 net = layers.dropout(
#                     net, dropout_keep_prob, is_training=trainable, scope='dropout6')
#                 net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
#                 net = layers.dropout(
#                     net, dropout_keep_prob, is_training=trainable, scope='dropout7')
#                 net = layers.conv2d(net, 1, [1, 1], scope='fc8')
#                 # Predict roll, pitch, yaw, translation_yaw, translation_pitch
#                 # activation from (-1,1), multiply with pi to get angles ranging from -pi to pi
#
#                 net = tf.squeeze(net, [1, 3])  # should have shape (batch_size,26)
#
#             # Predict roll, pitch, yaw, translation_yaw, translation_pitch
#             # activation from (-1,1), multiply with pi to get angles ranging from -pi to pi
#             motion_angles = slim.fully_connected(net, 5, scope='predict_fun',
#                                                  activation_fn=nn.tanh)
#
#             pi = 3.14159265358979323846
#             motion_angles = tf.scalar_mul(pi, motion_angles)
#
#             return motion_angles


def funnet(flow, trainable=False):
    _, height, width, _ = flow.shape.as_list()

    with tf.variable_scope('funnet') as sc:
        weight_decay = 0.0005
        with slim.arg_scope(alexnet_v2_arg_scope(weight_decay)):
            net, end_points = alexnet_v2(flow, num_classes=5, is_training=trainable, global_pool=False,
                                         spatial_squeeze=False)
            bs, height, width, channels = net.shape.as_list()
            net = tf.reshape(net, (bs, height * width * channels))
            add_to_summary('debug/alexnet/output', net)
        # print(end_points)

        with slim.arg_scope([slim.fully_connected],
                            biases_initializer=tf.constant_initializer(0.1),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.005),
                            outputs_collections="motion_angles"):
            motion_angles = slim.fully_connected(net, 5, activation_fn=tf.nn.tanh, scope="fc_final")
            # add_to_summary('debug/funnet/output', net)
            pi = 3.14159265358979323846
            motion_angles = tf.scalar_mul(pi, motion_angles)

            return motion_angles


def funnet_loss(motion_angle_prediction, flow, intrinsics):
    # Weight loss in function of flow amplitude.
    # For small flow, fundamental error is always small (norm(F) goes to zero for translation going to zero)
    # Calculate squared norm of flow
    # flow2 = tf.multiply(flow, flow)
    # weight = tf.reduce_mean(tf.reduce_sum(flow2, axis=2))
    weight = 1.
    # Several flow layers are outputted at different resolution.
    # First two are always du and dv at highest res.
    # Epipolar error of flow
    predict_fun = get_fundamental_matrix(motion_angle_prediction, intrinsics)
    print("get epipolar error")
    loss = math_ops.reduce_mean(tf.clip_by_value(tf.abs(epipolar_errors(predict_fun, flow)), 0., 100.))
    print("done epipolar error")
    # Weight loss
    loss = tf.scalar_mul(weight, loss)
    tf.losses.add_loss(loss)

    # Return the 'total' loss: loss fns + regularization terms defined in the model
    return tf.losses.get_total_loss()
