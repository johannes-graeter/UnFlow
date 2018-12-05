import os

import tensorflow as tf

import eval_gui
from e2eflow.gui import display
from e2eflow.kitti.data import KITTIData
from e2eflow.kitti.input import KITTIInput
from e2eflow.util import config_dict

FLAGS = eval_gui.FLAGS

import cv2
import numpy as np

from e2eflow.core.util import get_translation_rotation


def track_points(flows, first_image):
    """
    Track points in an image sequence as in the KLT tracker but with learned flow.
    :param flows: estimated flow, shape(sequ_length,height,width,2)
    :param images: input images for corner extraciton shape(sequ_length,height,width,3)
    :return: tracklets with tracked feature points; np.array, with shape(sequ_length, maxCorners,2).
            Tracklet length is always smaller/equal sequence length
    """

    # Caclucate points of interest in first image.
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Points has shape (num_features, 1, 2)
    points = cv2.goodFeaturesToTrack(cv2.cvtColor(first_image, cv2.COLOR_RGB2GRAY), mask=None, **feature_params)
    points = np.squeeze(np.array(points))  # shape (num_features,2)

    out = [points]

    for flow in flows:
        dudv = []
        for u, v in points.astype(int):
            du = 0
            dv = 0
            if 0 < u < first_image.shape[1] and 0 < v < first_image.shape[0]:
                du, dv = flow[v, u]
            dudv.append((du, dv))
        dudv = np.array(dudv)
        points = points + dudv
        out.append(points)
        # print(points)

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
    ts, rots = get_translation_rotation(motion_angles)
    assert (ts.shape[0] == rots.shape[0])
    batch_size = ts.shape[0]
    out = np.zeros((batch_size, 4, 4))
    for i in range(batch_size):
        out[i, :3, :3] = rots[i, :, :]
        out[i, :3, 3] = ts[i, :]
        out[i, 3, 3] = 1.
    return out


def accumulate_motion(motions):
    accumulator = np.eye(4)

    out = [accumulator]
    for m in motions:
        out.append(accumulator.dot(to_affine(m)))

    return out


def draw_trajectory(acc_motion, res=(200, 200)):
    img = np.ones((acc_motion.shape[0], res[0], res[1]))
    out = []
    for m in acc_motion:
        p = (m[0, 3], -m[2, 3])
        img = cv2.circle(img, p, 4, (0., 0., 0.), 2)
        dir = m.dot(np.array([0., 0., 1., 1.]))
        p2 = (dir[0, 3], -dir[2, 3])
        img = cv2.line(img, p, p2, (1.,0.,0.),2)
        out.append(img)

    return out


def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    print("-- evaluating: on {} pairs from {}/{}"
          .format(FLAGS.num, FLAGS.dataset, FLAGS.variant))

    default_config = config_dict()
    dirs = default_config['dirs']

    if FLAGS.dataset == 'kitti':
        data = KITTIData(dirs['data'], development=True)
        data_input = KITTIInput(data, batch_size=1, normalize=False,
                                # dims=(320, 1152))
                                dims=(384, 1280))

    input_fn0 = getattr(data_input, 'input_raw')
    input_fn = lambda: input_fn0(needs_crop=True, center_crop=True, seed=None, swap_images=False)

    results = []
    import numpy as np
    for name in FLAGS.ex.split(','):
        # Here we get eval images, names and the estimated flow per iteration.
        # This should be sequences.
        display_images, image_names, flows = eval_gui._evaluate_experiment(name, input_fn, data_input)

        # Draw image with tracked features.
        flows = np.squeeze(np.array(flows), axis=1)
        imgs = np.squeeze(np.array(display_images), axis=2)
        first_imgs = imgs[:, -1, :, :]
        tracked_points = track_points(flows, first_imgs[0, :, :, :])
        track_imgs = draw_tracked_points(tracked_points, first_imgs)
        imgs[:, -1, :, :] = track_imgs
        imgs = np.expand_dims(imgs, axis=2)
        image_names[-1] = "tracklets"

        # Accumulate and draw motion
        # Make dummy motion
        motions = np.zeros((flows.shape[0], 5))
        motions[:, 2] = 0.05

        traj_imgs = draw_trajectory(accumulate_motion(motions))
        for img in traj_imgs:
            cv2.imshow("traj", img)
            cv2.waitKey(0)

        results.append(imgs)

    display(results, image_names)


if __name__ == '__main__':
    tf.app.run()
