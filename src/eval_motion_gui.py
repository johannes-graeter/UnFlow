import eval_gui
from e2eflow.kitti.data import KITTIDataOdometry

FLAGS = eval_gui.FLAGS
from eval_gui import *

import cv2
import numpy as np
import sys

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
    print(imgs.shape)
    first_imgs = imgs[:, -2, 0, :, :]  # im1 for whole sequence
    points = get_keypoints(first_imgs[0, :, :, :], params['feature_params'])
    tracked_points = track_points(flows, points)
    track_imgs = draw_tracked_points(tracked_points, first_imgs)
    imgs[:, -2, 0, :, :] = track_imgs

    return imgs


def dump_images(dir, data_names, iterations):
    data, names = data_names

    for i, images in zip(iterations, data):
        for img, n in zip(images, names):
            dirname = dir + "/" + n
            try:
                os.makedirs(dirname)
            except:
                pass
            img_write = np.squeeze(img, axis=0) * 255.
            cv2.imwrite("{}/{}.png".format(dirname, i), img_write)


def dump_motion(dir, motions, iterations):
    try:
        os.makedirs(dir + "/motion_angles/")
    except:
        pass
    for i, motion in zip(iterations, motions):
        np.savetxt(dir + "/motion_angles/{}.txt".format(i), motion)


def do_plotting(display_images_out, motion_angles, image_names, motion_dim, fw_bw):
    display_images = []
    for l in display_images_out:
        prob_mask = l[3][motion_dim][fw_bw][:, :, :, 1]  # First object forward flow.
        display_images.append([l[0], l[1], l[2], np.stack((prob_mask, prob_mask, prob_mask), axis=3)])

    # Add empty image for motion plotting
    for l in display_images:
        l.append(np.ones_like(l[0]))

    # Convert to numpy
    imgs = np.array(display_images, copy=False, subok=True)

    # # Draw image with tracked features.
    # params = {
    #     'feature_params': dict(maxCorners=1000, qualityLevel=0.3, minDistance=10, blockSize=7),
    #     'resample_rate': 10}
    # imgs = add_flow_to_display(imgs, flows, params)
    # image_names[-1] = "tracklets"

    # motion is from current to last, so direction of translation is negative.
    scales = np.ones((motion_angles.shape[0])) * (-0.5)
    imgs = add_motion_to_display(imgs, motion_angles, scales)

    for i in range(imgs.shape[0]):
        imgs[i, -2, :, :, :] = imgs[i, -2, :, :, :] - imgs[i, -2, :, :, :].min()
        imgs[i, -2, :, :, :] = imgs[i, -2, :, :, :] / imgs[i, -2, :, :, :].max()

    return imgs, image_names


def evaluate_experiment2(name, input_fn, data_input, num_steps, start_iter):
    config_path, params, config, ckpt, ckpt_path = get_checkpoint(name)

    num_iters = start_iter
    max_iter = FLAGS.num if FLAGS.num > 0 else None

    with tf.Graph().as_default():  # , tf.device('gpu:' + FLAGS.gpu):
        inputs = input_fn()
        im1, im2, input_shape, intrinsics = inputs[:4]

        height, width, _ = tf.unstack(tf.squeeze(input_shape), num=3, axis=0)

        _, flow, flow_bw, motion_angles_tf, inlier_prob_mask = unsupervised_loss(
            (im1, im2, input_shape, intrinsics),
            normalization=data_input.get_normalization(),
            params=params, augment_photometric=False, return_flow=True, use_8point=False)

        flow_fw_int16 = flow_to_int16(flow)
        flow_bw_int16 = flow_to_int16(flow_bw)

        im1_pred = image_warp(im2, flow)
        im1_diff = tf.abs(im1 - im1_pred)

        image_slots = [(im1 / 255, 'first image'),
                       # (im1_pred / 255, 'warped second image', 0, 1),
                       (im1_diff / 255, 'warp error'),
                       # (im2 / 255, 'second image', 1, 0),
                       # (im2_diff / 255, '|first - second|', 1, 2),
                       (flow_to_color(flow), 'flow prediction'),
                       (inlier_prob_mask, 'inlier_prob'),
                       (im1 / 255, 'motion')
                       ]

        num_ims = len(image_slots)
        image_ops = [t[0] for t in image_slots]
        image_names = [t[1] for t in image_slots]

        sess_config = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            restore_networks(sess, params, ckpt, ckpt_path)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            image_lists = []
            iterations = []

            # JG: return flow
            flows = []
            motion_angles = []
            while not coord.should_stop() and (num_iters <= num_steps + start_iter):
                all_results = sess.run(
                    [flow, flow_bw, flow_fw_int16, flow_bw_int16, motion_angles_tf] + image_ops)
                # JG: get flow
                flows.append(all_results[0])
                # JG: get motion
                motion_angles.append(all_results[4])

                # flow_fw_res, flow_bw_res, flow_fw_int16_res, flow_bw_int16_res = all_results[:4]
                all_results = all_results[5:]
                image_results = all_results[:num_ims]
                image_lists.append(image_results)
                iterations.append(num_iters)

                sys.stdout.write('\r')
                num_iters += 1
                sys.stdout.write("-- evaluating '{}': {}/{}".format(name, num_iters, max_iter))
                sys.stdout.flush()
                print()

            coord.request_stop()
            coord.join(threads)

    return image_lists, motion_angles, image_names, iterations


def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    print("-- evaluating: on {} pairs from {}/{}"
          .format(FLAGS.num, 'kitti_odometry', 'test_set'))

    default_config = config_dict()
    dirs = default_config['dirs']
    print("-- loading from {}".format(dirs))

    input_dims = (320, 1152)

    if FLAGS.dataset == 'kitti':
        data = KITTIDataOdometry(dirs['data_testing'], development=False, do_fetch=False)
        data_input = KITTIInput(data, batch_size=1, normalize=False, dims=input_dims, num_threads=1)
    else:
        raise Exception("Motion eval only implemented for KITTI yet!")

    input_fn0 = getattr(data_input, 'input_raw')

    # shift = np.random.randint(0, 1e6)  # get different set of images each call
    # print("shift by {} images".format(shift))

    # shift = 215441

    def input_fn(iter):
        return input_fn0(augment_crop=False, center_crop=True, seed=None, swap_images=False, shift=iter)

    results = []
    motion_dim = 1
    fw_bw = 0

    start_iter = 0
    num_steps = 50

    dump_images = False

    for n, name in enumerate(FLAGS.ex.split(',')):
        # Here we get eval images, names and the estimated flow per iteration.
        # This should be sequences.
        # Motion angles are roll, pitch,yaw, translation_yaw, translation ptich from current image to last image
        # eval_gui.evaluate_experiment(name, input_fn, data_input, do_resize=False)
        dumped = False
        try:
            while True:
                print("start_iter", start_iter)
                image_lists, motion_angles, image_names, iterations = evaluate_experiment2(name, lambda: input_fn(
                    -start_iter), data_input, num_steps, start_iter)

                # Extract motion corresponding to ego motion.
                for i in range(len(motion_angles)):
                    motion_angles[i] = motion_angles[i][motion_dim][fw_bw]
                motion_angles = np.squeeze(np.array(motion_angles), axis=1)

                dump_motion("/tmp/UnFlow_results/", motion_angles, iterations)

                if dump_images:
                    plot_res = do_plotting(image_lists, motion_angles, image_names, motion_dim=motion_dim, fw_bw=fw_bw)
                    dump_images("/tmp/UnFlow_results/", plot_res, iterations)

                if not dumped:
                    data_input.dump_names("/tmp/UnFlow_results/filenames.txt")
                    dumped = True
                start_iter += num_steps

        except tf.errors.OutOfRangeError as exc:
            print(exc.message)
            print("Out of range thrown, returning")
            pass


if __name__ == '__main__':
    tf.app.run()
