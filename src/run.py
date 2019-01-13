import copy
import os

import tensorflow as tf

from e2eflow.cityscapes.data import CityscapesData
from e2eflow.cityscapes.input import CityscapesInput
from e2eflow.core.train import Trainer
from e2eflow.experiment import Experiment
from e2eflow.kitti.data import KITTIDataOdometry, KITTIDataRaw
from e2eflow.kitti.input import KITTIInput
from e2eflow.util import convert_input_strings

tf.app.flags.DEFINE_string('ex', 'default',
                           'Name of the experiment.'
                           'If the experiment folder already exists in the log dir, '
                           'training will be continued from the latest checkpoint.')
tf.app.flags.DEFINE_boolean('debug', False,
                            'Enable image summaries and disable checkpoint writing for debugging.')
tf.app.flags.DEFINE_boolean('ow', False,
                            'Overwrites a previous experiment with the same name (if present)'
                            'instead of attempting to continue from its latest checkpoint.')
FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    experiment = Experiment(
        name=FLAGS.ex,
        overwrite=FLAGS.ow)
    dirs = experiment.config['dirs']
    run_config = experiment.config['run']

    gpu_list_param = run_config['gpu_list']

    if isinstance(gpu_list_param, int):
        gpu_list = [gpu_list_param]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_list_param)
    else:
        gpu_list = list(range(len(gpu_list_param.split(','))))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list_param
    gpu_batch_size = int(run_config['batch_size'] / max(len(gpu_list), 1))
    devices = ['/gpu:' + str(gpu_num) for gpu_num in gpu_list]

    train_dataset = run_config.get('dataset', 'kitti_raw')

    edata = KITTIDataOdometry(data_dir=dirs['data_testing'], fast_dir=dirs.get('fast'), stat_log_dir=None,
                              development=run_config['development'], do_fetch=False)
    einput = KITTIInput(data=edata, batch_size=1, normalize=False)

    print('Read training data from {}'.format(dirs['data_training']))

    if 'kitti' in train_dataset:
        if train_dataset == 'kitti_raw':
            data = KITTIDataRaw(data_dir=dirs['data_training'], fast_dir=dirs.get('fast'), stat_log_dir=None,
                                development=run_config['development'], do_fetch=False)
        elif train_dataset == 'kitti_odometry':
            data = KITTIDataOdometry(data_dir=dirs['data_training'], fast_dir=dirs.get('fast'), stat_log_dir=None,
                                     development=run_config['development'], do_fetch=False)
        else:
            raise Exception("Dataset {} is unknown".format(train_dataset))
        config = copy.deepcopy(experiment.config['train'])
        config.update(experiment.config['train_kitti'])
        convert_input_strings(config, dirs)
        iters = config.get('num_iters', 0)
        input = KITTIInput(data=data,
                           batch_size=gpu_batch_size,
                           normalize=False,
                           dims=(config['height'], config['width']))
    elif train_dataset == 'cityscapes':
        data = CityscapesData(data_dir=dirs['data_training'], sub_dir="train", fast_dir=dirs.get('fast'),
                              stat_log_dir=None,
                              development=run_config['development'])
        config = copy.deepcopy(experiment.config['train'])
        config.update(experiment.config['train_cityscapes'])
        convert_input_strings(config, dirs)
        iters = config.get('num_iters', 0)

        input = CityscapesInput(data=data,
                                batch_size=gpu_batch_size,
                                normalize=False,
                                dims=(config['height'], config['width']))
    else:
        raise Exception("Only Kitti supported.")
    tr = Trainer(
        lambda shift: input.input_raw(swap_images=False,
                                      augment_crop=False,
                                      center_crop=True,
                                      shift=shift * run_config['batch_size']),
        lambda: einput.input_raw(swap_images=False,
                                 augment_crop=False,
                                 center_crop=True),
        params=config,
        normalization=input.get_normalization(),
        train_summaries_dir=experiment.train_dir,
        eval_summaries_dir=experiment.eval_dir,
        experiment=FLAGS.ex,
        ckpt_dir=experiment.save_dir,
        debug=FLAGS.debug,
        interactive_plot=run_config.get('interactive_plot'),
        devices=devices)
    tr.run(0, iters)
    if not FLAGS.debug:
        experiment.conclude()


# if train_dataset == 'chairs':
#     raise Exception("Motion trianing not yet implemented.")
#     cconfig = copy.deepcopy(experiment.config['train'])
#     cconfig.update(experiment.config['train_chairs'])
#     convert_input_strings(cconfig, dirs)
#     citers = cconfig.get('num_iters', 0)
#     cdata = ChairsData(data_dir=dirs['data'],
#                        fast_dir=dirs.get('fast'),
#                        stat_log_dir=None,
#                        development=run_config['development'])
#     cinput = ChairsInput(data=cdata,
#              batch_size=gpu_batch_size,
#              normalize=False,
#              dims=(cconfig['height'], cconfig['width']))
#     tr = Trainer(
#           lambda shift: cinput.input_raw(swap_images=False,
#                                          shift=shift * run_config['batch_size']),
#           lambda: einput.input_train_2012(),
#           params=cconfig,
#           normalization=cinput.get_normalization(),
#           train_summaries_dir=experiment.train_dir,
#           eval_summaries_dir=experiment.eval_dir,
#           experiment=FLAGS.ex,
#           ckpt_dir=experiment.save_dir,
#           debug=FLAGS.debug,
#           interactive_plot=run_config.get('interactive_plot'),
#           devices=devices)
#     tr.run(0, citers)
#
# elif train_dataset == 'kitti':
#     kconfig = copy.deepcopy(experiment.config['train'])
#     kconfig.update(experiment.config['train_kitti'])
#     convert_input_strings(kconfig, dirs)
#     kiters = kconfig.get('num_iters', 0)
#     kinput = KITTIInput(data=kdata,
#                         batch_size=gpu_batch_size,
#                         normalize=False,
#                         dims=(kconfig['height'], kconfig['width']))
#     tr = Trainer(
#           lambda shift: kinput.input_raw(swap_images=False,
#                                          center_crop=True,
#                                          shift=shift * run_config['batch_size']),
#           lambda: einput.input_train_2012(),
#           params=kconfig,
#           normalization=kinput.get_normalization(),
#           train_summaries_dir=experiment.train_dir,
#           eval_summaries_dir=experiment.eval_dir,
#           experiment=FLAGS.ex,
#           ckpt_dir=experiment.save_dir,
#           debug=FLAGS.debug,
#           interactive_plot=run_config.get('interactive_plot'),
#           devices=devices)
#     tr.run(0, kiters)
#
# elif train_dataset == 'cityscapes':
#     raise Exception("Motion trianing not yet implemented.")
#     kconfig = copy.deepcopy(experiment.config['train'])
#     kconfig.update(experiment.config['train_cityscapes'])
#     convert_input_strings(kconfig, dirs)
#     kiters = kconfig.get('num_iters', 0)
#     cdata = CityscapesData(data_dir=dirs['data'],
#                 fast_dir=dirs.get('fast'),
#                 stat_log_dir=None,
#                 development=run_config['development'])
#     kinput = KITTIInput(data=cdata,
#                         batch_size=gpu_batch_size,
#                         normalize=False,
#                         dims=(kconfig['height'], kconfig['width']))
#     tr = Trainer(
#           lambda shift: kinput.input_raw(swap_images=False,
#                                          center_crop=True,
#                                          skip=[0, 1],
#                                          shift=shift * run_config['batch_size']),
#           lambda: einput.input_train_2012(),
#           params=kconfig,
#           normalization=kinput.get_normalization(),
#           train_summaries_dir=experiment.train_dir,
#           eval_summaries_dir=experiment.eval_dir,
#           experiment=FLAGS.ex,
#           ckpt_dir=experiment.save_dir,
#           debug=FLAGS.debug,
#           interactive_plot=run_config.get('interactive_plot'),
#           devices=devices)
#     tr.run(0, kiters)
#
# elif train_dataset == 'synthia':
#     raise Exception("Motion trianing not yet implemented.")
#     sconfig = copy.deepcopy(experiment.config['train'])
#     sconfig.update(experiment.config['train_synthia'])
#     convert_input_strings(sconfig, dirs)
#     siters = sconfig.get('num_iters', 0)
#     sdata = SynthiaData(data_dir=dirs['data'],
#             fast_dir=dirs.get('fast'),
#             stat_log_dir=None,
#             development=run_config['development'])
#     sinput = KITTIInput(data=sdata,
#                         batch_size=gpu_batch_size,
#                         normalize=False,
#                         dims=(sconfig['height'], sconfig['width']))
#     tr = Trainer(
#           lambda shift: sinput.input_raw(swap_images=False,
#                                          shift=shift * run_config['batch_size']),
#           lambda: einput.input_train_2012(),
#           params=sconfig,
#           normalization=sinput.get_normalization(),
#           train_summaries_dir=experiment.train_dir,
#           eval_summaries_dir=experiment.eval_dir,
#           experiment=FLAGS.ex,
#           ckpt_dir=experiment.save_dir,
#           debug=FLAGS.debug,
#           interactive_plot=run_config.get('interactive_plot'),
#           devices=devices)
#     tr.run(0, siters)
#
# elif train_dataset == 'kitti_ft':
#     raise Exception("Motion trianing not yet implemented.")
#     ftconfig = copy.deepcopy(experiment.config['train'])
#     ftconfig.update(experiment.config['train_kitti_ft'])
#     convert_input_strings(ftconfig, dirs)
#     ftiters = ftconfig.get('num_iters', 0)
#     ftinput = KITTIInput(data=kdata,
#                          batch_size=gpu_batch_size,
#                          normalize=False,
#                          dims=(ftconfig['height'], ftconfig['width']))
#     tr = Trainer(
#           lambda shift: ftinput.input_train_gt(40),
#           lambda: einput.input_train_2015(40),
#           supervised=True,
#           params=ftconfig,
#           normalization=ftinput.get_normalization(),
#           train_summaries_dir=experiment.train_dir,
#           eval_summaries_dir=experiment.eval_dir,
#           experiment=FLAGS.ex,
#           ckpt_dir=experiment.save_dir,
#           debug=FLAGS.debug,
#           interactive_plot=run_config.get('interactive_plot'),
#           devices=devices)
#     tr.run(0, ftiters)
#
# else:
#   raise ValueError(
#       "Invalid dataset. Dataset must be one of "
#       "{synthia, kitti, kitti_ft, cityscapes, chairs}")

#    if not FLAGS.debug:
#        experiment.conclude()

if __name__ == '__main__':
    tf.app.run()
