import tensorflow as tf

from .augment import random_photometric
from .downsample import downsample
from .flow_util import flow_to_color
from .flownet import flownet, FLOW_SCALE
from .funnet import funnet, get_funnet_log_uncertainties
from .image_warp import image_warp
from .losses import compute_losses, create_border_mask, funnet_loss, compute_exp_reg_loss
from .util import add_to_output, get_reference_explain_mask, \
    get_mask_fundamental_mat

# REGISTER ALL POSSIBLE LOSS TERMS
LOSSES = ['occ', 'sym', 'fb', 'grad', 'ternary', 'photo', 'smooth_1st', 'smooth_2nd']


def maybe_update_max_elements(ref, percentage=0.05, dim=1, min_thres=0.3):
    b, h, w, c = ref.shape.as_list()
    cur = ref[:, :, :, dim]

    # Get threshold by highest values.
    fl = tf.contrib.layers.flatten(cur)  # get flattened tensors (batch_size, :)
    _, length = fl.shape.as_list()
    thres = tf.nn.top_k(fl, k=int(percentage * length), sorted=True)
    thres = tf.expand_dims(thres.values[:, -1], axis=1)

    # Clip threshold, so that values that are a lot lower will not be updated.
    thres = tf.clip_by_value(thres, clip_value_min=min_thres, clip_value_max=1.0)

    # Get mask which is true if value is greater than thres.
    gr_mask_stack = tf.reshape(tf.reshape(tf.ones_like(cur), (b, h * w)) * thres, (b, h, w))
    mask = tf.greater(cur, gr_mask_stack)
    mask = tf.stack((mask, mask), axis=3)

    # Update.
    ref = tf.where(mask, get_reference_explain_mask(ref.shape.as_list()), ref)
    return ref


def _track_loss(op, name):
    tf.add_to_collection('losses', tf.identity(op, name=name))


def _track_image(op, name, namespace="train", normalize=False):
    name = namespace + '/' + name

    if normalize:
        op = op - tf.reduce_min(op)
        op = tf.div_no_nan(op, tf.reduce_max(op))

    if len(op.shape.as_list()) < 4:
        op = tf.expand_dims(op, axis=3)

    tf.add_to_collection('train_images', tf.identity(op, name=name))


def unsupervised_loss(batch, params, normalization=None, augment_photometric=True,
                      return_flow=False):
    channel_mean = tf.constant(normalization[0]) / 255.0
    im1, im2, _, intrin = batch
    im1 = im1 / 255.0
    im2 = im2 / 255.0
    im_shape = tf.shape(im1)[1:3]

    # -------------------------------------------------------------------------
    # Data & mask augmentation
    border_mask = create_border_mask(im1, 0.1)

    # _track_image(im1, 'orig1')
    # _track_image(im2, 'orig2')

    if augment_photometric:
        im1_photo, im2_photo = random_photometric([im1, im2],
                                                  noise_stddev=0.04, min_contrast=-0.3, max_contrast=0.3,
                                                  brightness_stddev=0.02, min_colour=0.9, max_colour=1.1,
                                                  min_gamma=0.7, max_gamma=1.5)
    else:
        im1_photo, im2_photo = im1, im2

    _track_image(im1_photo, 'augmented1')
    _track_image(im2_photo, 'augmented2')

    # Images for neural network input with mean-zero values in [-1, 1]
    im1_photo = im1_photo - channel_mean
    im2_photo = im2_photo - channel_mean

    flownet_spec = params.get('flownet', 'S')
    full_resolution = params.get('full_res', False)
    assert (full_resolution is False)
    train_all = params.get('train_all')

    flows_fw, flows_bw = flownet(im1_photo, im2_photo,
                                 flownet_spec=flownet_spec,
                                 full_resolution=full_resolution,
                                 backward_flow=True,
                                 train_all=train_all)

    flows_fw = flows_fw[-1]
    flows_bw = flows_bw[-1]

    # -------------------------------------------------------------------------
    # Losses
    layer_weights = [12.7, 4.35, 3.9, 3.4, 1.1]
    layer_patch_distances = [3, 2, 2, 1, 1]
    if full_resolution:
        layer_weights = [12.7, 5.5, 5.0, 4.35, 3.9, 3.4, 1.1]
        layer_patch_distances = [3, 3] + layer_patch_distances
        im1_s = im1
        im2_s = im2
        mask_s = border_mask
        final_flow_scale = FLOW_SCALE * 4
        final_flow_fw = flows_fw[0] * final_flow_scale
        final_flow_bw = flows_bw[0] * final_flow_scale
    else:
        im1_s = downsample(im1, 4)
        im2_s = downsample(im2, 4)
        mask_s = downsample(border_mask, 4)
        final_flow_scale = FLOW_SCALE
        final_flow_fw = tf.image.resize_bilinear(flows_fw[0], im_shape) * final_flow_scale * 4
        final_flow_bw = tf.image.resize_bilinear(flows_bw[0], im_shape) * final_flow_scale * 4

    combined_losses = dict()
    combined_loss = 0.0
    for loss in LOSSES:
        combined_losses[loss] = 0.0

    if params.get('pyramid_loss'):
        flow_enum = enumerate(zip(flows_fw, flows_bw))
    else:
        flow_enum = [(0, (flows_fw[0], flows_bw[0]))]

    for i, flow_pair in flow_enum:
        layer_name = "loss" + str(i + 2)

        flow_scale = final_flow_scale / (2 ** i)

        with tf.variable_scope(layer_name):
            layer_weight = layer_weights[i]
            flow_fw_s, flow_bw_s = flow_pair

            mask_occlusion = params.get('mask_occlusion', '')
            assert mask_occlusion in ['fb', 'disocc', '']

            losses = compute_losses(im1_s, im2_s,
                                    flow_fw_s * flow_scale, flow_bw_s * flow_scale,
                                    border_mask=mask_s if params.get('border_mask') else None,
                                    mask_occlusion=mask_occlusion,
                                    data_max_distance=layer_patch_distances[i])

            layer_loss = 0.0

            for loss in LOSSES:
                weight_name = loss + '_weight'
                if params.get(weight_name):
                    _track_loss(losses[loss], loss)
                    layer_loss += params[weight_name] * losses[loss]
                    combined_losses[loss] += layer_weight * losses[loss]

            combined_loss += layer_weight * layer_loss

            im1_s = downsample(im1_s, 2)
            im2_s = downsample(im2_s, 2)
            mask_s = downsample(mask_s, 2)

    motions = []
    masks = []
    fun_losses = []
    mask_consistency_losses = []

    # Debug
    refs = []

    # Start with reference mask with ones.
    # ref = get_reference_explain_mask(flows_fw[0].shape.as_list())
    # inlier_thres = 0.6  # For full size image.
    # inlier_thres = 0.03
    inlier_thres = 0.001
    ref = get_mask_fundamental_mat(flows_fw[0], inlier_thres=inlier_thres, number_iterations=50)
    # ref = get_mask_fundamental_mat(final_flow_fw, inlier_thres=inlier_thres, number_iterations=5)
    # ref = tf.image.resize_bilinear(ref, tf.shape(flows_fw[0])[1:3])

    num_objects = 2
    for i in range(num_objects):
        # Add loss from epipolar geometry for forward pass.
        motion_angles, mask_logits = funnet(flows_fw[0], ref[:, :, :, 1])
        # Convert mask of logits to inlier probability. Upscale for flow weighting. Same method as for upscaling final_flow_fw.
        probs = tf.nn.softmax(mask_logits)
        inlier_probs_full_res = tf.image.resize_bilinear(tf.expand_dims(probs[:, :, :, 1], axis=3), im_shape)
        fun_loss = funnet_loss(motion_angles, final_flow_fw, inlier_probs_full_res, intrin)
        # Regularize to pull all inlier probs towards 1.
        fw_mask_loss = compute_exp_reg_loss(pred=mask_logits, ref=ref)

        # Add loss from epipolar geometry for backward pass (more training data).
        motion_angles_bw, mask_logits_bw = funnet(flows_bw[0], probs[:, :, :, 1])  # uses auto_reuse
        probs_bw = tf.nn.softmax(mask_logits_bw)
        inlier_probs_bw_full_res = tf.image.resize_bilinear(tf.expand_dims(probs_bw[:, :, :, 1], axis=3), im_shape)
        fun_loss_bw = funnet_loss(motion_angles_bw, final_flow_bw, inlier_probs_bw_full_res, intrin)

        # Regularize backward inlier mask to be very similar to forward mask.
        warped_bw_logits = image_warp(mask_logits_bw, flows_fw[0])
        bw_mask_loss = compute_exp_reg_loss(pred=warped_bw_logits, ref=probs)

        motions.append((motion_angles, motion_angles_bw))
        masks.append((probs, probs_bw))
        fun_losses.append((fun_loss, fun_loss_bw))
        mask_consistency_losses.append((fw_mask_loss, bw_mask_loss))
        refs.append(ref)

        # Next reference is mean probability, but inverted (outlier prob beomces object inlier prob)
        # ref = tf.nn.softmax(warped_bw_logits)

        # Reverse last dim to make outliers inliers.
        ref = tf.stack((ref[:, :, :, 1], ref[:, :, :, 0]), axis=3)
        # # Recalculate mask with epipolar constraint.
        # ref_rescale = tf.image.resize_bilinear(tf.expand_dims(ref[:, :, :, 1], axis=3), im_shape)
        #
        # _track_image(ref_rescale, "/inv_rescale_ref_{}".format(i), namespace="funnet")
        #
        # ref = get_mask_fundamental_mat(final_flow_fw, inlier_thres=inlier_thres, inlier_probs=ref_rescale,
        #                                number_iterations=1)
        # ref = tf.image.resize_bilinear(ref, tf.shape(flows_fw[0])[1:3])

        # Set a percentage of biggest pixels to 1 inorder to regularize to 1.
        # ref = maybe_update_max_elements(ref, percentage=0.05, min_thres=0.3)

    # All mask probabilities accumulated must give for each pixel
    # Accumulate masks
    accumulated_mask_fw = tf.zeros_like(ref[:, :, :, 1])
    accumulated_mask_bw = tf.zeros_like(ref[:, :, :, 1])
    for mask_fw, mask_bw in masks:
        accumulated_mask_fw = accumulated_mask_fw + mask_fw[:, :, :, 1]
        accumulated_mask_bw = accumulated_mask_bw + mask_bw[:, :, :, 1]
    # Mean squared difference to 1. is loss.
    weight_global_mask_loss = 10000.
    global_mask_consistency_loss = (
        weight_global_mask_loss * tf.reduce_mean(tf.square(tf.ones_like(accumulated_mask_fw) - accumulated_mask_fw)),
        weight_global_mask_loss * tf.reduce_mean(tf.square(tf.ones_like(accumulated_mask_bw) - accumulated_mask_bw)))

    # Add up all local mask consistency losses
    local_mask_consistency_loss = [0., 0.]
    # Different weight for static and dynamic, make trainable?
    obj_weights = [1., 3.]
    assert (len(obj_weights) == num_objects)
    for w, (ml_fw, ml_bw) in zip(obj_weights, mask_consistency_losses):
        local_mask_consistency_loss[0] += w * ml_fw
        local_mask_consistency_loss[1] += w * ml_bw

    funnet_log_unc = get_funnet_log_uncertainties(size=num_objects + 2)

    # Add losses from funnet to problem.
    if params.get('train_motion_only'):
        regularization_loss = tf.losses.get_regularization_loss(scope="funnet")
        final_loss = regularization_loss

        assert (len(fun_losses) == num_objects)
        # Add loss for motion.
        for i, l in enumerate(fun_losses):
            final_loss += tf.scalar_mul(0.5 * tf.exp(-funnet_log_unc[i]), l[0] + l[1])

        # Add loss for global mask consistency -> all probabilities add up to one.
        final_loss += tf.scalar_mul(0.5 * tf.exp(-funnet_log_unc[-2]),
                                    global_mask_consistency_loss[0] + global_mask_consistency_loss[1])

        # Add loss for local mask constitency -> output resembles input.
        final_loss += tf.scalar_mul(0.5 * tf.exp(-funnet_log_unc[-1]),
                                    local_mask_consistency_loss[0] + local_mask_consistency_loss[1])

        # Add regularization for all weights.
        for i in range(funnet_log_unc.shape.as_list()[0]):
            final_loss += tf.scalar_mul(0.5, funnet_log_unc[i])

    else:
        raise Exception("Not implemented flow estimation with motion yet.")
        # regularization_loss = tf.losses.get_regularization_loss()
        # final_loss = regularization_loss + tf.exp(-weight_flow) * combined_loss + weight_flow\
        #              + tf.exp(-weight_fw) * fun_loss + tf.exp(-weight_fw_mask) * fw_mask_loss + weight_fw + weight_fw_mask \
        #              #+ tf.exp(-weight_bw) * fun_loss_bw + tf.exp(-weight_bw_mask) * bw_mask_loss + weight_bw + weight_bw_mask

    masks_out = []
    for m in masks:
        masks_out.append([tf.image.resize_bilinear(m[0][:, :, :, :], im_shape),
                         tf.image.resize_bilinear(m[1][:, :, :, :], im_shape)])

    ##################################
    #  DEBUG
    ##################################
    # Debug
    for j, mo in enumerate(motions):
        for i in range(5):
            add_to_output('funnet/motion_angles_{}/{}'.format(j, i), mo[0][:, i])
            add_to_output('funnet/motion_angles_bw_{}/{}'.format(j, i), mo[1][:, i])

    for i, l in enumerate(fun_losses):
        _track_loss(l[0], 'loss/fun_loss_{}'.format(i))
        _track_loss(l[1], 'loss/fun_loss_bw_{}'.format(i))

    for i, m in enumerate(masks):
        _track_image(m[0][:, :, :, 1], 'mask_full_{}'.format(i), namespace='funnet')
        _track_image(m[1][:, :, :, 1], 'mask_full_bw_{}'.format(i), namespace='funnet')

    for i, r in enumerate(refs):
        _track_image(r[:, :, :, 1], "ref_{}".format(i), namespace='funnet')

    _track_loss(regularization_loss, 'loss/reg_nets')
    _track_loss(combined_loss, 'loss/variable_loss')

    # Add regularization loss of mask to problem.
    _track_loss(global_mask_consistency_loss[0], 'loss/reg_global_mask_fw')
    _track_loss(global_mask_consistency_loss[1], 'loss/reg_global_mask_bw')
    _track_loss(local_mask_consistency_loss[0], 'loss/reg_local_mask_fw')
    _track_loss(local_mask_consistency_loss[1], 'loss/reg_local_mask_bw')

    for i in range(num_objects):
        _track_loss(funnet_log_unc[i], 'loss/unc_fun_loss_{}'.format(i))
    _track_loss(funnet_log_unc[-2], 'loss/unc_mask_global')
    _track_loss(funnet_log_unc[-1], 'loss/unc_mask_local')
    _track_loss(final_loss, 'loss/final')

    for loss in LOSSES:
        _track_loss(combined_losses[loss], 'loss/' + loss)
        weight_name = loss + '_weight'
        if params.get(weight_name):
            weight = tf.identity(params[weight_name], name='weight/' + loss)
            tf.add_to_collection('params', weight)

    _track_image(flow_to_color(final_flow_fw), 'estimated_flow_fw')

    im1_pred = image_warp(im2, final_flow_fw)
    _track_image(im1_pred, 'warp_2to1')
    _track_image(tf.abs(im1 - im1_pred) / 255, 'diff')

    ##################################
    #  End of DEBUG
    ##################################
    if not return_flow:
        return final_loss

    return final_loss, final_flow_fw, final_flow_bw, motions, masks_out
