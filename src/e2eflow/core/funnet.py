import tensorflow as tf

from .funnet_architectures import custom_frontend, custom_frontend_resnet, exp_mask_layers, trunc_normal

slim = tf.contrib.slim


def funnet(flow):
    def frontend(input_flow, scope):
        """Define frontend to use."""
        # return alexnet_v2(input_flow, num_classes=None, spatial_squeeze=False, scope=scope)
        return custom_frontend_resnet(input_flow, scope=scope)

    with tf.variable_scope('funnet') as sc:
        # Frontend
        # Get flow feature map from fully convolutional frontend.
        conv_activations, end_points = frontend(flow, scope=sc.original_name_scope)

        # Mask layers
        mask, end_points_mask = exp_mask_layers(conv_activations, flow, 2, scope=sc.original_name_scope)
        end_points.update(end_points_mask)

        # Backend
        net = conv_activations[-1]

        with slim.arg_scope([slim.fully_connected],
                            biases_initializer=tf.constant_initializer(0.001),
                            weights_regularizer=slim.l2_regularizer(0.05),
                            weights_initializer=trunc_normal(0.1),
                            outputs_collections="motion_angles"):

            # Reduce information for fully connected.
            # Use conv2d instead of fully_connected layers.
            net = slim.conv2d(net, 1024, [1, 1], scope='fc7')
            # end_points[sc.name + '/fc7'] = net

            net = slim.conv2d(net, 10, [1, 1], scope='fc8')
            # end_points[sc.name + '/fc8'] = net

            # Reshape for fully connected net.
            bs, height, width, channels = net.shape.as_list()
            net = tf.reshape(net, (bs, height * width * channels))

            # Learn motion from feature map (net).
            # Should have reasonable height and width to preserve spatial information.
            motion_angles = slim.fully_connected(net, 5, activation_fn=tf.nn.tanh, scope="fc_final")

            # add_to_summary('debug/funnet/output', net)
            pi = 3.14159265358979323846
            motion_angles = tf.scalar_mul(pi / 2., motion_angles)

            return motion_angles, mask
