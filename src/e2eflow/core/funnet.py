import tensorflow as tf

from .funnet_architectures import custom_frontend, default_arg_scope, trunc_normal
from .util import add_to_debug_output

slim = tf.contrib.slim


def funnet(flow):
    def frontend(input_flow, scope):
        """Define frontend to use."""
        # return alexnet_v2(input_flow, num_classes=None, spatial_squeeze=False, scope=scope)
        return custom_frontend(input_flow, scope=scope)

    with tf.variable_scope('funnet') as sc:
        weight_decay = 0.0005

        # Frontend
        with slim.arg_scope(default_arg_scope(weight_decay)):
            # Get flow fetaure map from fully convolutional frontend.
            net, end_points = frontend(flow, scope=sc.original_name_scope)

            # Reduce information for fully connected.
            net = slim.conv2d(net, 10, [1, 1], weights_initializer=trunc_normal(0.005), scope='fc8')
            end_points[sc.name + '/fc8'] = net

            add_to_debug_output('debug/alexnet/output', net)

        # Backend
        with slim.arg_scope([slim.fully_connected],
                            biases_initializer=tf.constant_initializer(0.0001),  # very small bias
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.05),  # ca. 10 degrees std dev
                            outputs_collections="motion_angles"):
            # Reshape for fully connected net.
            bs, height, width, channels = net.shape.as_list()
            net = tf.reshape(net, (bs, height * width * channels))

            # Learn motion from feature map (net).
            # Should have reasonable height and width to preserve spatial information.
            motion_angles = slim.fully_connected(net, 5, activation_fn=tf.nn.tanh, scope="fc_final")

            # add_to_summary('debug/funnet/output', net)
            pi = 3.14159265358979323846
            motion_angles = tf.scalar_mul(pi, motion_angles)

            return motion_angles
