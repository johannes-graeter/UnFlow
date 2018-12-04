import os

import tensorflow as tf

import eval_gui
from e2eflow.gui import display
from e2eflow.kitti.data import KITTIData
from e2eflow.kitti.input import KITTIInput
from e2eflow.util import config_dict

FLAGS = eval_gui.FLAGS


def draw_tracked_points(flow1, image1):
    raise Exception("Implement")


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
                                #dims=(320, 1152))
                                dims=(384, 1280))

    input_fn0 = getattr(data_input, 'input_raw')
    input_fn = lambda: input_fn0(needs_crop=True, center_crop=True, seed=None)

    results = []
    for name in FLAGS.ex.split(','):
        # Here we get eval images, names and the estimated flow per iteration.
        # This should be sequences.
        display_images, image_names, flows = eval_gui._evaluate_experiment(name, input_fn, data_input)

        #tracked_points = track_points(flows[0], display_images[-1])
        # motion_top_view = draw_motion(motion)
        results.append(display_images)

    display(results, image_names)


if __name__ == '__main__':
    tf.app.run()
