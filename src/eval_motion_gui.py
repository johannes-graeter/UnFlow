import os

import tensorflow as tf

import eval_gui
from e2eflow.gui import display
from e2eflow.kitti.data import KITTIDataOdometry
from e2eflow.kitti.input import KITTIInput
from e2eflow.util import config_dict

FLAGS = eval_gui.FLAGS

import cv2
import numpy as np

from e2eflow.core.util import get_translation_rotation

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

# params for ShiTomasi corner detection
feature_params_def = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)


def track_points(flows, points):
    """
    Track points in an image sequence as in the KLT tracker but with learned flow.
    :param flows: estimated flow, shape(sequ_length,height,width,2)
    :param images: input images for corner extraciton shape(sequ_length,height,width,3)
    :return: tracklets with tracked feature points; np.array, with shape(sequ_length, maxCorners,2).
            Tracklet length is always smaller/equal sequence length
    """

    out = [points]

    for flow in flows:
        dudv = []
        for u, v in points.astype(int):
            du = 0
            dv = 0
            if 0 < u < flow.shape[1] and 0 < v < flow.shape[0]:
                du, dv = flow[v, u]
            dudv.append((du, dv))
        dudv = np.array(dudv)
        points = points + dudv
        out.append(points)

    return out


def draw_tracked_points(tracked_points, images):
    max_num_points = max([len(tp) for tp in tracked_points])
    colors = np.random.randint(0, 255, (max_num_points, 3))

    if not images.dtype == int:
        colors = colors / 255.
    # draw the tracks
    output = images.copy()
    mask = np.zeros_like(images[0])
    for i in range(1, len(tracked_points) - 1):
        concat = np.concatenate((tracked_points[i - 1], tracked_points[i]), axis=1)
        for num, (u_last, v_last, u, v) in enumerate(concat.astype(int)):
            mask = cv2.line(mask, (u_last, v_last), (u, v), colors[num, :].tolist(), 2)
            output[i] = cv2.circle(output[i], (u, v), 4, colors[num, :].tolist(), 2)
        output[i] = cv2.add(output[i], mask)
    return output


def to_affine(motion_angles):
    """transform from rpy, trans_yaw, trans_pitch to affine transform"""
    ts, rots = get_translation_rotation(tf.constant(motion_angles))
    with tf.Session() as sess:
        ts = np.squeeze(ts.eval(), axis=2)
        rots = rots.eval()
    assert (ts.shape[0] == rots.shape[0])
    batch_size = ts.shape[0]
    out = np.zeros((batch_size, 4, 4))
    for i in range(batch_size):
        out[i, :3, :3] = rots[i, :, :]
        out[i, :3, 3] = ts[i, :]
        out[i, 3, 3] = 1.
    return out


def accumulate_motion(motions, scales, invert=False):
    accumulator = np.eye(4)

    out = [accumulator]
    for m, s in zip(to_affine(motions), scales):
        # Apply scale
        m[:3, 3] = m[:3, 3] / np.linalg.norm(m[:3, 3]) * s
        # Invert if necessary
        if invert:
            m = np.linalg.inv(m)
        # Accumulate
        out.append(out[-1].dot(m))
    # First element is double
    out.pop(0)
    return out


def draw_trajectory(acc_motion, min_res=(300, 300)):
    fig = Figure(figsize=(3, 3))  # size in inches: 1 inch=2.54 cm
    canvas = FigureCanvas(fig)
    width, height = [int(x) for x in fig.get_size_inches() * fig.get_dpi()]
    ax = fig.gca()

    def get_dims(acc_motion, border_perc=0.1):
        x_max = acc_motion[:, 0, 3].max()
        x_min = acc_motion[:, 0, 3].min()

        z_max = acc_motion[:, 2, 3].max()
        z_min = acc_motion[:, 2, 3].min()

        dmax = max(x_max - x_min, z_max - z_min)
        x_max = x_min + dmax
        z_max = z_min + dmax
        x_min -= dmax / 2
        z_min -= dmax / 2

        border = dmax * border_perc

        return (x_min - border, x_max + border), (z_min - border, z_max + border)

    x_minmax, z_minmax = get_dims(acc_motion)
    # imgs = np.ones((acc_motion.shape[0], res[0], res[1], 3), dtype=float)
    out = []
    for m in acc_motion:
        artists = []
        p = (m[0, 3], m[2, 3])
        artists.append(plt.Circle(p, 0.1, color='b'))
        dir = m[:3, :3].dot(np.array([0., 0., 1.]))
        p2 = (dir[0], dir[2])
        artists.append(plt.Arrow(p[0], p[1], p2[0], p2[1]))

        for a in artists:
            ax.add_artist(a)

        ax.axis("equal")
        ax.set_xlim(x_minmax)
        ax.set_ylim(z_minmax)

        canvas.draw()  # draw the canvas, cache the renderer
        img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape((width, height, 3))
        img = cv2.resize(img, dsize=min_res)
        img = img / 255.
        out.append(img)

    return out


def resize_with_pad(image, height, width, pad_color=(0., 0., 0.)):
    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)

    resized_image = cv2.resize(constant, (height, width))

    return resized_image


def draw_angles_as_text(imgs, motions):
    # Add info about yaw and pitch
    dim_names = ['roll', 'pitch', 'yaw', 'trans_yaw', 'trans_pitch']
    for i in range(len(imgs)):
        m = motions[i]
        text = ['{0}={1:.2e}'.format(t, v) for t, v in zip(dim_names, m)]
        dy = 40
        y = 10
        for l in text:
            y += dy
            imgs[i, -1, 0] = cv2.putText(imgs[i, -1, 0], l, (40, y), 0, 0.8, (0, 0, 0), 2)
    return imgs


def add_motion_to_display(imgs, motions, scales):
    height1 = imgs.shape[3]
    width1 = imgs.shape[4]

    traj_imgs = np.array(
        draw_trajectory(np.array(accumulate_motion(motions, scales=scales, invert=True)), min_res=(height1, height1)))
    height0 = traj_imgs.shape[1]
    width0 = traj_imgs.shape[2]

    dh = (height1 - height0) // 2
    dw = (width1 - width0) // 2
    assert (dh >= 0 and dw >= 0)

    traj_imgs = np.expand_dims(traj_imgs, axis=1)

    imgs[:, -1, :, dh:height0 + dh, dw:width0 + dw] = traj_imgs

    imgs = draw_angles_as_text(imgs, motions)
    return imgs


def get_keypoints(image, feature_params):
    # Caclucate points of interest in first image.
    # Points has shape (num_features, 1, 2)
    points = cv2.goodFeaturesToTrack(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), mask=None, **feature_params)
    points = np.squeeze(np.array(points))  # shape (num_features,2)
    return points


def add_flow_to_display(imgs, flows, params):
    first_imgs = imgs[:, -2, 0, :, :]  # im1 for whole sequence
    points = get_keypoints(first_imgs[0, :, :, :], params['feature_params'])
    tracked_points = track_points(flows, points)
    track_imgs = draw_tracked_points(tracked_points, first_imgs)
    imgs[:, -2, 0, :, :] = track_imgs

    return imgs


def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    print("-- evaluating: on {} pairs from {}/{}"
          .format(FLAGS.num, FLAGS.dataset, FLAGS.variant))

    default_config = config_dict()
    dirs = default_config['dirs']
    print("-- loading from {}".format(dirs))

    input_dims = (320, 1152)

    if FLAGS.dataset == 'kitti':
        data = KITTIDataOdometry(dirs['data_testing'], development=False, do_fetch=False)
        data_input = KITTIInput(data, batch_size=1, normalize=False, dims=input_dims, num_threads=1)
    else:
        raise Exception("Motion eval only implemented for KITTI yet!")

    while True:
        input_fn0 = getattr(data_input, 'input_raw')
        shift = np.random.randint(0, 1e6)  # get different set of images each call
        print("shift by {} images".format(shift))

        def input_fn():
            return input_fn0(augment_crop=False, center_crop=True, seed=None, swap_images=False, shift=shift)

        results = []
        for n, name in enumerate(FLAGS.ex.split(',')):
            # Here we get eval images, names and the estimated flow per iteration.
            # This should be sequences.
            # Motion angles are roll, pitch,yaw, translation_yaw, translation ptich from current image to last image
            display_images, image_names, flows, motion_angles = eval_gui.evaluate_experiment(name, input_fn, data_input,
                                                                                             do_resize=False)
            flows = np.squeeze(np.array(flows), axis=1)

            # Add ones for motion plotting and convert to numpy
            for l in display_images:
                l.append(np.ones_like(l[0]))
            imgs = np.array(display_images, copy=False, subok=True)

            # Draw image with tracked features.
            params = {
                'feature_params': dict(maxCorners=1000, qualityLevel=0.3, minDistance=10, blockSize=7),
                'resample_rate': 10}
            imgs = add_flow_to_display(imgs, flows, params)
            image_names[-1] = "tracklets"

            # Accumulate and draw motion
            motion_angles = np.squeeze(np.array(motion_angles), axis=1)

            # motion is from current to last, so direction of translation is negative.
            scales = np.ones((motion_angles.shape[0])) * (-0.5)
            imgs = add_motion_to_display(imgs, motion_angles, scales)
            image_names.append("motion")

            results.append(imgs)

        display(results, image_names)


if __name__ == '__main__':
    tf.app.run()
