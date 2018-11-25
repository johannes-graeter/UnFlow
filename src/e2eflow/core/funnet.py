import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

from .util import epipolar_errors
from .util import get_fundamental_matrix
from .util import pad

slim = tf.contrib.slim


def _leaky_relu(x):
    with tf.variable_scope('leaky_relu'):
        return tf.maximum(0.1 * x, x)


def funnet(flow, trainable=False):
    """

    :param flow: input tensor with shape(batch_size, height, width, 2)
    :param trainable: True if should be trained
    :return: motion prediction of shape(batch_size, 5) with angles roll, pitch, yaw, trans_yaw, trans_pitch
            all in range(-pi/2, pi/2)
    """
    # _, height, width, _ = flow.shape.as_list()
    with tf.variable_scope('FunNet'):
        # concat_inputs = tf.concat([inputs['input_a'],
        #                           inputs['input_b'],
        #                           inputs['brightness_error_sd'],
        #                           inputs['brightness_error_css'],
        #                           inputs['flow']], axis=3)
        with slim.arg_scope([slim.conv2d],
                            # Only backprop this network if trainable
                            trainable=trainable,
                            data_format='NCHW',
                            weights_regularizer=slim.l2_regularizer(0.0004),
                            weights_initializer=layers.variance_scaling_initializer(),
                            activation_fn=_leaky_relu
                            ):
            with slim.arg_scope([slim.conv2d], stride=2):
                # conv_1 = slim.conv2d(pad(concat_inputs, 3), 64, 7, scope='conv1')
                conv_1 = slim.conv2d(pad(flow, 3), 64, 7, scope='conv1')
                conv_2 = slim.conv2d(pad(conv_1, 2), 128, 5, scope='conv2')
                conv_3 = slim.conv2d(pad(conv_2, 2), 256, 5, scope='conv3')

            conv3_1 = slim.conv2d(pad(conv_3), 256, 3, scope='conv3_1')
            with slim.arg_scope([slim.conv2d], num_outputs=512, kernel_size=3):
                conv4 = slim.conv2d(pad(conv3_1), stride=2, scope='conv4')
                conv4_1 = slim.conv2d(pad(conv4), scope='conv4_1')
                conv5 = slim.conv2d(pad(conv4_1), stride=2, scope='conv5')
                conv5_1 = slim.conv2d(pad(conv5), scope='conv5_1')
            conv6 = slim.conv2d(pad(conv5_1), 1024, 3, stride=2, scope='conv6')
            conv6_1 = slim.conv2d(pad(conv6), 1024, 3, scope='conv6_1')

            conv6_2 = slim.conv2d(pad(conv6_1), 1, 3,
                                  scope='conv6_2')
            conv6_2 = tf.squeeze(conv6_2, [1, 3])  # should have shape (batch_size,26)

            # Predict roll, pitch, yaw, translation_yaw, translation_pitch
            # activation from (-1,1), multiply with pi to get angles ranging from -pi to pi
            motion_angles = slim.fully_connected(conv6_2, 5, scope='predict_fun',
                                                 activation_fn=nn.tanh)
            pi = 3.14159265358979323846
            motion_angles = tf.scalar_mul(pi, motion_angles)

            return motion_angles


def funnet_loss(motion_angle_prediction, flow, intrinsics):
    # Weight loss in function of flow amplitude.
    # For small flow, fundamental error is always small (norm(F) goes to zero for translation going to zero)
    # Calculate squared norm of flow
    weight = math_ops.reduce_mean(tf.multiply(flow, flow))

    # Several flow layers are outputted at different resolution.
    # First two are always du and dv at highest res.
    batch_size, height, width, num_flows = flow.shape.as_list()

    # flow_thres = 3.
    # if weight > flow_thres:

    # Epipolar error of flow
    predict_fun = get_fundamental_matrix(motion_angle_prediction, intrinsics)
    print("get epipolar error")
    loss = math_ops.reduce_mean(epipolar_errors(tf.reshape(predict_fun, (batch_size, 9, 1)), flow))
    print("done epipolar error")
    # Weight loss
    loss = tf.scalar_mul(weight, loss)
    tf.losses.add_loss(loss)

    # Return the 'total' loss: loss fns + regularization terms defined in the model
    return tf.losses.get_total_loss()
