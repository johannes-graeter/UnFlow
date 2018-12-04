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


def track_points(flows, images):
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
    points = cv2.goodFeaturesToTrack(cv2.cvtColor(images[0, :, :, :], cv2.COLOR_RGB2GRAY), mask=None, **feature_params)
    points = np.squeeze(np.array(points))  # shape (num_features,2)

    out = [points]

    for flow in flows:
        dudv = []
        for u, v in points.astype(int):
            du, dv = flow[v, u]
            dudv.append((du, dv))
        dudv = np.array(dudv)
        points = points - dudv
        out.append(points)
        # print(points)

    return out


def draw_tracked_points(tracked_points, image1):
    colors = np.random.randint(0, 255, (len(tracked_points[0]), 3))
    # draw the tracks
    mask = np.zeros_like(image1)
    for i in range(1, len(tracked_points) - 1):
        concat = np.concatenate((tracked_points[i - 1], tracked_points[i]), axis=1)
        for num, (u_last, v_last, u, v) in enumerate(concat):
            mask = cv2.line(mask, (u_last, v_last), (u, v), colors[num].tolist(), 1)
            image1 = cv2.circle(image1, (u, v), 1, colors[num].tolist(), -1)
    img = cv2.add(image1, mask)
    return img


def draw_motion(motion):
    raise Exception("Implement")


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

        # Get flow for tracking features.
        flows = np.squeeze(np.array(flows), axis=1)

        # Get first images for corner extraction.
        imgs = np.squeeze(np.array(display_images[-1]), axis=1)

        tracked_points = track_points(flows, imgs)
        img = draw_tracked_points(tracked_points, imgs[0, :, :])


        # Todo: integrate in gui
        from matplotlib import pyplot as plt
        plt.imshow(img)
        display_images.extend(img)

        image_names.extend("tracklets")

        results.append(display_images)

    display(results, image_names)


if __name__ == '__main__':
    tf.app.run()
