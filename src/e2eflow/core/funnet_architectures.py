import tensorflow as tf

slim = tf.contrib.slim


def _leaky_relu(x):
    with tf.variable_scope('leaky_relu'):
        return tf.maximum(0.1 * x, x)


def trunc_normal(stddev):
    return tf.truncated_normal_initializer(0.0, stddev)


def custom_frontend(inputs,
                    scope='custom_frontend'):
    with tf.variable_scope(scope, 'custom_frontend', [inputs]) as sc:
        end_points_collection = sc.original_name_scope  # + '_end_points'

        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            net = slim.conv2d(inputs, 16, [7, 7], stride=2, scope='cnv1')
            net = slim.conv2d(net, 32, [5, 5], stride=2, scope='cnv2')
            net = slim.conv2d(net, 64, [3, 3], stride=2, scope='cnv3')
            net = slim.conv2d(net, 128, [3, 3], stride=2, scope='cnv4')
            net = slim.conv2d(net, 256, [3, 3], stride=1, scope='cnv5')
            print(net.shape.as_list())
            # net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

            # Use conv2d instead of fully_connected layers.
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=trunc_normal(0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                net = slim.conv2d(net, 1024, [1, 1], scope='fc7')
                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)
                end_points[sc.name + '/fc8'] = net
    return net, end_points


def alexnet_v2(inputs,
               num_classes=1000,
               spatial_squeeze=True,
               scope='alexnet_v2'):
    """AlexNet version 2.
    Described in: http://arxiv.org/pdf/1404.5997v2.pdf
    Parameters from:
    github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
    layers-imagenet-1gpu.cfg
    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224 or set
          global_pool=True. To use in fully convolutional mode, set
          spatial_squeeze to false.
          The LRN layers have been removed and change the initializers from
          random_normal_initializer to xavier_initializer.
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: the number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer are returned instead.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        logits. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.
      global_pool: Optional boolean flag. If True, the input to the classification
        layer is avgpooled to size 1x1, for any input size. (This is not part
        of the original AlexNet.)
    Returns:
      net: the output of the logits layer (if num_classes is a non-zero integer),
        or the non-dropped-out input to the logits layer (if num_classes is 0
        or None).
      end_points: a dict of tensors with intermediate activations.
    """
    with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
        end_points_collection = sc.original_name_scope  # + '_end_points'

        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
                              scope='conv1')
            # net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
            net = slim.conv2d(net, 192, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            net = slim.conv2d(net, 384, [3, 3], scope='conv3')
            net = slim.conv2d(net, 384, [3, 3], scope='conv4')
            net = slim.conv2d(net, 256, [3, 3], scope='conv5')
            # net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

            # Use conv2d instead of fully_connected layers.
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=trunc_normal(0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                #        net = slim.conv2d(net, 4096, [5, 5], padding='VALID',
                #                          scope='fc6')
                net = slim.conv2d(net, 1024, [1, 1], scope='fc7')
                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)
                if num_classes:
                    net = slim.conv2d(net, num_classes, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      biases_initializer=tf.zeros_initializer(),
                                      scope='fc8')
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    end_points[sc.name + '/fc8'] = net
    return net, end_points


def default_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        biases_initializer=tf.constant_initializer(0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        outputs_collections='funnet'):
        with slim.arg_scope([slim.conv2d], padding='SAME'):
            with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                return arg_sc


def motion_net_lowe(flow):
    """Inspired by openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_Unsupervised_Learning_of_CVPR_2017_paper.pdf"""

    def pose_exp_net(flow_fw, flow_bw=None, do_exp=True, scope='pose_exp_net'):
        """Inspired by openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_Unsupervised_Learning_of_CVPR_2017_paper.pdf
        Modifications: Different input, 5 motion parameters, tanh for activations (since we want to map to 5 angles).
        Original:
        Input:  src_image_stack: images from which motion should start
                tgt_image: image where motino should end
                do_exp: if true, mask is calcualted with weights for outliers.
        Ours:
        Input:  flow_fw: flow before upsampling with shape (batch_size, height, width, 2) channels correspond to du and dv
                flow_bw: flow before upsampling with shape (batch_size, height, width, 2) channels correspond to du and dv
                do_exp: if true, mask is calcualted with weights for outliers.
        Output:
                motion_pred: prediction of angels: roll,ptich,yaw,trans_yaw, trans_pitch; shape=(batch_size,5)
                masks:?
        """
        # inputs = tf.concat([tgt_image, src_image_stack], axis=3)
        if flow_bw:
            inputs = tf.concat([flow_fw, flow_bw], axis=3)
            mask_channels = 2 * 2
        else:
            inputs = flow_fw
            mask_channels = 2 * 1

        # H = inputs.get_shape()[1].value
        # W = inputs.get_shape()[2].value
        # num_source = int(src_image_stack.get_shape()[3].value // 3)
        with tf.variable_scope(scope, 'pose_exp_net', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                normalizer_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.05),
                                activation_fn=tf.nn.relu,
                                outputs_collections=end_points_collection):
                # cnv1 to cnv5b are shared between pose and explainability prediction
                cnv1 = slim.conv2d(inputs, 16, [7, 7], stride=2, scope='cnv1')
                cnv2 = slim.conv2d(cnv1, 32, [5, 5], stride=2, scope='cnv2')
                cnv3 = slim.conv2d(cnv2, 64, [3, 3], stride=2, scope='cnv3')
                cnv4 = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
                cnv5 = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
                # Pose specific layers
                with tf.variable_scope('motion_angles'):
                    cnv6 = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
                    cnv7 = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                    motion_pred = slim.conv2d(cnv7, 5 * 1, [1, 1], scope='pred',
                                              stride=1, normalizer_fn=None, activation_fn=tf.nn.tanh)
                    motion_pred = tf.reduce_mean(motion_pred, [1, 2])
                # Exp mask specific layers
                if do_exp:
                    print("get weight mask")
                    with tf.variable_scope('exp'):
                        upcnv5 = slim.conv2d_transpose(cnv5, 256, [3, 3], stride=2, scope='upcnv5')

                        upcnv4 = slim.conv2d_transpose(upcnv5, 128, [3, 3], stride=2, scope='upcnv4')
                        mask4 = slim.conv2d(upcnv4, mask_channels, [3, 3], stride=1, scope='mask4',
                                            normalizer_fn=None, activation_fn=None)

                        upcnv3 = slim.conv2d_transpose(upcnv4, 64, [3, 3], stride=2, scope='upcnv3')
                        mask3 = slim.conv2d(upcnv3, mask_channels, [3, 3], stride=1, scope='mask3',
                                            normalizer_fn=None, activation_fn=None)

                        upcnv2 = slim.conv2d_transpose(upcnv3, 32, [5, 5], stride=2, scope='upcnv2')
                        mask2 = slim.conv2d(upcnv2, mask_channels, [5, 5], stride=1, scope='mask2',
                                            normalizer_fn=None, activation_fn=None)

                        upcnv1 = slim.conv2d_transpose(upcnv2, 16, [7, 7], stride=2, scope='upcnv1')
                        mask1 = slim.conv2d(upcnv1, mask_channels, [7, 7], stride=1, scope='mask1',
                                            normalizer_fn=None, activation_fn=None)
                else:
                    mask1 = None
                    mask2 = None
                    mask3 = None
                    mask4 = None
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                # if do_exp:
                # return motion_pred, [mask1, mask2, mask3, mask4], end_points
                # else:
                return motion_pred

    with tf.variable_scope('funnet') as sc:
        motion_angles = pose_exp_net(flow, flow_bw=None, do_exp=False, scope=sc.original_name_scope)
        # add_to_summary('debug/funnet/output', net)
        pi = 3.14159265358979323846
        motion_angles = tf.scalar_mul(pi, motion_angles)

        return motion_angles
