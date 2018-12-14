import tensorflow as tf
from tensorflow.python.ops import math_ops

from .alexnet import alexnet_v2, alexnet_v2_arg_scope
from .util import add_to_debug_output
from .util import epipolar_errors
from .util import get_fundamental_matrix

slim = tf.contrib.slim


def _leaky_relu(x):
    with tf.variable_scope('leaky_relu'):
        return tf.maximum(0.1 * x, x)


def funnet(flow):
    with tf.variable_scope('funnet') as sc:
        weight_decay = 0.0005
        with slim.arg_scope(alexnet_v2_arg_scope(weight_decay)):
            # Num classes
            net, end_points = alexnet_v2(flow, num_classes=5, spatial_squeeze=False)
            bs, height, width, channels = net.shape.as_list()
            net = tf.reshape(net, (bs, height * width * channels))
            add_to_debug_output('debug/alexnet/output', net)

        with slim.arg_scope([slim.fully_connected],
                            biases_initializer=tf.constant_initializer(0.0001),  # very small bias
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.05),  # ca. 10 degrees std dev
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

    # loss = math_ops.reduce_mean(tf.clip_by_value(tf.abs(epipolar_errors(predict_fun, flow)), 0., 100.))
    loss = math_ops.reduce_mean(epipolar_errors(predict_fun, flow, normalize=True, debug=True))

    # Add loss
    tf.losses.add_loss(tf.scalar_mul(weight, loss))

    # Return the 'total' loss: loss fns + regularization terms defined in the model
    return tf.losses.get_total_loss()
