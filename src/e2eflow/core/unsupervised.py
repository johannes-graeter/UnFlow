import tensorflow as tf

from .augment import random_affine, random_photometric
from .downsample import downsample
from .flow_util import flow_to_color
from .flownet import flownet, FLOW_SCALE
from .funnet import funnet
from .image_warp import image_warp
from .losses import compute_losses, create_border_mask, funnet_loss, compute_exp_reg_loss
from .util import add_to_debug_output, get_reference_explain_mask, get_inlier_prob_from_mask_logits

# REGISTER ALL POSSIBLE LOSS TERMS
LOSSES = ['occ', 'sym', 'fb', 'grad', 'ternary', 'photo', 'smooth_1st', 'smooth_2nd']


def _track_loss(op, name):
    tf.add_to_collection('losses', tf.identity(op, name=name))


def _track_image(op, name):
    name = 'train/' + name
    tf.add_to_collection('train_images', tf.identity(op, name=name))


def unsupervised_loss(batch, params, normalization=None, augment=False,
                      return_flow=False):
    channel_mean = tf.constant(normalization[0]) / 255.0
    im1, im2, _, intrin = batch
    im1 = im1 / 255.0
    im2 = im2 / 255.0
    im_shape = tf.shape(im1)[1:3]

    # -------------------------------------------------------------------------
    # Data & mask augmentation
    border_mask = create_border_mask(im1, 0.1)

    _track_image(im1, 'orig1')
    _track_image(im2, 'orig2')

    if augment:
        raise Exception("Implement intrinsics from augmentation")
        im1_geo, im2_geo, border_mask_global = random_affine(
            [im1, im2, border_mask],
            horizontal_flipping=True,
            min_scale=0.9, max_scale=1.1
        )

        # augment locally
        im2_geo, border_mask_local = random_affine(
            [im2_geo, border_mask],
            min_scale=0.9, max_scale=1.1
        )
        border_mask = border_mask_local * border_mask_global

        im1_photo, im2_photo = random_photometric(
            [im1_geo, im2_geo],
            noise_stddev=0.04, min_contrast=-0.3, max_contrast=0.3,
            brightness_stddev=0.02, min_colour=0.9, max_colour=1.1,
            min_gamma=0.7, max_gamma=1.5)

        _track_image(im1_photo, 'augmented1')
        _track_image(im2_photo, 'augmented2')
    else:
        im1_geo, im2_geo = im1, im2
        im1_photo, im2_photo = im1, im2

    # Images for loss comparisons with values in [0, 1] (scale to original using * 255)
    im1_norm = im1_geo
    im2_norm = im2_geo
    # Images for neural network input with mean-zero values in [-1, 1]
    im1_photo = im1_photo - channel_mean
    im2_photo = im2_photo - channel_mean

    flownet_spec = params.get('flownet', 'S')
    full_resolution = params.get('full_res')
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
        im1_s = im1_norm
        im2_s = im2_norm
        mask_s = border_mask
        final_flow_scale = FLOW_SCALE * 4
        final_flow_fw = flows_fw[0] * final_flow_scale
        final_flow_bw = flows_bw[0] * final_flow_scale
    else:
        im1_s = downsample(im1_norm, 4)
        im2_s = downsample(im2_norm, 4)
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

    # Add loss from epipolar geometry
    motion_angles, mask_logits = funnet(flows_fw[0])

    # Get regularization for explanation mask.
    # If we want several motions, ref_exp_mask will be the exp mask from the motion before.
    ref_exp_mask = get_reference_explain_mask(mask_logits.shape.as_list())
    reg_loss_exp_mask = compute_exp_reg_loss(cur_exp_logits, ref_exp_mask)

    # Convert mask of logits to inlier probability.
    inlier_probs = get_inlier_prob_from_mask_logits(mask_logits)

    # intrin = tf.Print(intrin, ["intrinsics", intrin], summarize=100)
    fun_loss = funnet_loss(motion_angles, final_flow_fw, inlier_probs, intrin)

    # Debug
    for i in range(5):
        add_to_debug_output('funnet/motion_angles/{}'.format(i), motion_angles[:, i])
    add_to_debug_output('funnet/final_flow', final_flow_fw)
    add_to_debug_output('funnet/input', flows_fw[0])
    add_to_debug_output('funnet/loss', fun_loss)
    _track_image('funnet/mask', inlier_probs)

    if params.get('train_motion_only'):
        combined_loss = params.get('epipolar_loss_weight') * fun_loss
        regularization_loss = tf.losses.get_regularization_loss(scope="funnet")
        regularization_loss += reg_loss_exp_mask
    else:
        combined_loss += params.get('epipolar_loss_weight') * fun_loss
        regularization_loss = tf.losses.get_regularization_loss()

    final_loss = combined_loss + regularization_loss
    _track_loss(final_loss, 'loss/combined')

    for loss in LOSSES:
        _track_loss(combined_losses[loss], 'loss/' + loss)
        weight_name = loss + '_weight'
        if params.get(weight_name):
            weight = tf.identity(params[weight_name], name='weight/' + loss)
            tf.add_to_collection('params', weight)

    for i, cur_flow in enumerate(flows_fw):
        _track_image(flow_to_color(cur_flow), 'flow_{}'.format(i))
    _track_image(flow_to_color(final_flow_fw), 'estimated_flow')

    im1_pred = image_warp(im2, final_flow_fw)
    _track_image(im1_pred, 'warp_2to1')
    _track_image(tf.abs(im1 - im1_pred) / 255, 'diff')

    if not return_flow:
        return final_loss

    return final_loss, final_flow_fw, final_flow_bw, motion_angles
