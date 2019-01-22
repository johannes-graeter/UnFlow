import os
import re
from multiprocessing import Process

import numpy as np
import tensorflow as tf

try:
    import memory_saving_gradients

    tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory
    # tf.__dict__["gradients"] = memory_saving_gradients.gradients_speed
    print("Use memory_saving_gradients reduce net memory usage for cost of speed. "
          "See https://github.com/openai/gradient-checkpointing.")
except ImportError:
    print("To fit bigger nets into memory get https://github.com/openai/gradient-checkpointing "
          "and put it in your Pythonpath.")
    pass

import tensorflow.contrib.slim as slim

from . import util
from .flow_util import flow_error_avg, flow_to_color, flow_error_image, outlier_pct
from .image_warp import image_warp
from .input import resize_output_crop
from .losses import occlusion, DISOCC_THRESH, create_outgoing_mask
from .supervised import supervised_loss
from .unsupervised import unsupervised_loss
# from .util import add_to_debug_output
from .util import summarized_placeholder
from ..gui import display
from ..ops import forward_warp


def restore_networks(sess, params, ckpt, ckpt_path=None):
    # Attention this is converted to checkpoints in e2eflow/util.py::convert_input_strings
    finetune = params.get('finetune', [])
    train_all = params.get('train_all', None)
    spec = params.get('flownet', 'S')
    flownet_num = len(spec)

    net_names = ['flownet_c'] + ['stack_{}_flownet'.format(i + 1) for i in range(flownet_num - 1)]
    if params.get('has_motion', True):
        net_names = net_names + ['funnet']

    assert len(finetune) <= flownet_num + 1
    # Save all trained networks, restore all networks which are kept fixed
    if train_all:
        restore_external_nets = finetune if ckpt is None else []
        variables_to_save = slim.get_variables_to_restore(include=net_names)
    else:
        # restore_external_nets = finetune if ckpt is None else finetune[:flownet_num - 1]
        restore_external_nets = finetune
        if params.get('epipolar_loss_weight', 0.) > 0. and not params.get('train_motion_only', False):
            print('-- save', net_names[-2:])
            variables_to_save = slim.get_variables_to_restore(include=net_names[-2:])
        else:
            print('-- save', net_names[-1])
            variables_to_save = slim.get_variables_to_restore(include=[net_names[-1]])

    saver = tf.train.Saver(variables_to_save, max_to_keep=1000)

    sess.run(tf.global_variables_initializer())

    for i, finetune_ckpt in enumerate(restore_external_nets):
        print('-- restore', net_names[i], finetune_ckpt.model_checkpoint_path)

        nets_to_restore = [net_names[i]]
        variables_to_restore = slim.get_variables_to_restore(
            include=nets_to_restore)

        try:
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, finetune_ckpt.model_checkpoint_path)
        except:
            # load partial network (missing final 2 upconvolutions)
            variables_to_restore = [v for v in variables_to_restore
                                    if not 'full_res' in v.name]
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, finetune_ckpt.model_checkpoint_path)

    # Restore currently training net always last, so that restored values maybe overriden.
    if ckpt is not None:
        # continue training
        print('-- restore and train ', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
    return saver


def _add_loss_summaries():
    losses = tf.get_collection('losses')
    for l in losses:
        tensor_name = re.sub('tower_[0-9]*/', '', l.op.name)
    losses = tf.get_collection('losses')
    for l in losses:
        tensor_name = re.sub('tower_[0-9]*/', '', l.op.name)
        tf.summary.scalar(tensor_name, l)


def _add_variable_summaries():
    ms = tf.get_collection('motion_angles')
    batch_size, var_length = ms[0].shape.as_list()
    for i in range(batch_size):
        for j in range(var_length):
            tensor_names = "motion_angles/batch{}/motion{}".format(i, j)
            tf.summary.scalar(tensor_names, ms[0][i, j])

    if len(ms) > 1:
        batch_size, var_length = ms[1].shape.as_list()
        for i in range(batch_size):
            for j in range(var_length):
                tensor_names = "motion_angles_bw/batch{}/motion{}".format(i, j)
                tf.summary.scalar(tensor_names, ms[1][i, j])


def _add_param_summaries():
    params = tf.get_collection('params')
    for p in params:
        tensor_name = re.sub('tower_[0-9]*/', '', p.op.name)
        tf.summary.scalar(tensor_name, p)


def _add_image_summaries():
    images = tf.get_collection('train_images')
    for im in images:
        tensor_name = re.sub('tower_[0-9]*/', '', im.op.name)
        tf.summary.image(tensor_name, im)


def _add_tensor_summaries(collection='debug_tensors'):
    ts = tf.get_collection(collection)
    for t in ts:
        name = re.sub('tower_[0-9]*/', '', t.op.name)
        tf.summary.scalar(name + '/mean', tf.reduce_mean(t))
        tf.summary.scalar(name + '/max', tf.reduce_max(t))
        tf.summary.scalar(name + '/min', tf.reduce_min(t))


def _eval_plot(results, image_names, title):
    display(results, image_names, title)


class Trainer:
    def __init__(self, train_batch_fn, eval_batch_fn, params,
                 train_summaries_dir, eval_summaries_dir, ckpt_dir,
                 normalization, debug=False, experiment="", interactive_plot=False,
                 supervised=False, devices=None):

        self.train_summaries_dir = train_summaries_dir
        self.eval_summaries_dir = eval_summaries_dir
        self.ckpt_dir = ckpt_dir
        self.params = params
        self.debug = debug
        self.train_batch_fn = train_batch_fn
        self.eval_batch_fn = eval_batch_fn
        self.normalization = normalization
        self.experiment = experiment
        self.interactive_plot = interactive_plot
        self.plot_proc = None
        self.supervised = supervised
        self.loss_fn = supervised_loss if supervised else unsupervised_loss
        self.devices = devices or '/gpu:0'
        self.shared_device = devices[0] if len(devices) == 1 else '/cpu:0'

    def run(self, min_iter, max_iter):
        """Train (at most) from min_iter + 1 to max_iter.
        If checkpoints are found in ckpt_dir,
        they must be have a global_step within [min_iter, max_iter]. In this case,
        training is continued from global_step + 1 until max_iter is reached.
        """
        save_interval = self.params['save_interval']

        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt is not None:
            ckpt_path = ckpt.model_checkpoint_path
            global_step = int(ckpt_path.split('/')[-1].split('-')[-1])
            assert global_step >= min_iter, 'training stage not reached'

            start_iter = global_step + 1
            if start_iter > max_iter:
                print('-- train: max_iter reached')
                return
        else:
            start_iter = min_iter + 1

        print('-- training from i = {} to {}'.format(start_iter, max_iter))

        # assert (max_iter - start_iter + 1) % save_interval == 0
        # for i in range(start_iter, max_iter + 1, save_interval):
        #     self.train(i, i + save_interval - 1, i - (min_iter + 1))
        #     # self.eval(1)
        i = start_iter
        while i < max_iter:
            current_iter = self.train(i, i + save_interval - 1, i - (min_iter + 1))
            i = current_iter + 1
            # # self.eval(1)

        if self.plot_proc:
            self.plot_proc.join()

    def get_train_and_loss_ops(self, batch, learning_rate, global_step):
        if self.params['flownet'] == 'resnet':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
        else:
            opt = tf.train.AdamOptimizer(beta1=0.9, beta2=0.999,
                                         learning_rate=learning_rate)

        def _add_summaries():
            _add_loss_summaries()
            _add_param_summaries()
            _add_variable_summaries()
            _add_tensor_summaries('tracked_tensors')
            if self.debug:
                _add_image_summaries()
                _add_tensor_summaries('debug_tensors')

        if len(self.devices) == 1:
            loss_ = self.loss_fn(batch, self.params, self.normalization)
            if self.params.get('train_motion_only'):
                scope = "funnet"
            else:
                scope = None
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            train_op = opt.minimize(loss_, var_list=train_vars)
            _add_summaries()
        else:
            tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i, devid in enumerate(self.devices):
                    with tf.device(devid):
                        with tf.name_scope('tower_{}'.format(i)) as scope:
                            loss_ = self.loss_fn(batch, self.params, self.normalization)
                            _add_summaries()

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

                            # Retain the summaries from the final tower.
                            tower_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES,
                                                                scope)
                            grads = opt.compute_gradients(loss_)
                            tower_grads.append(grads)

            grads = average_gradients(tower_grads)
            apply_gradient_op = opt.apply_gradients(grads)
            train_op = apply_gradient_op

        return train_op, loss_

    def train(self, start_iter, max_iter, iter_offset):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)

        with tf.Graph().as_default(), tf.device(self.shared_device):
            batch = self.train_batch_fn(iter_offset)

            with tf.name_scope('params') as scope:
                learning_rate_ = util.summarized_placeholder('learning_rate', 'train')
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            global_step_ = tf.placeholder(tf.int32, name="global_step")

            train_op, loss_ = self.get_train_and_loss_ops(batch, learning_rate_, global_step_)

            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
            summary_ = tf.summary.merge(summaries)

            sess_config = tf.ConfigProto(allow_soft_placement=True)

            with tf.Session(config=sess_config) as sess:
                if self.debug:
                    summary_writer = tf.summary.FileWriter(self.train_summaries_dir,
                                                           sess.graph)
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
                                                report_tensor_allocations_upon_oom=True)
                    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                else:
                    summary_writer = tf.summary.FileWriter(self.train_summaries_dir)
                    run_options = None
                    run_metadata = None

                saver = restore_networks(sess, self.params, ckpt)

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                save_iter = start_iter

                for local_i, i in enumerate(range(start_iter, max_iter + 1)):
                    # Stop if OutOfRangeException was raised.
                    if coord.should_stop():
                        break
                    # if INTERACTIVE_PLOT:
                    #    plt.title = "{} ({})".format(self.experiment, i)
                    decay_iters = local_i + iter_offset
                    if 'manual_decay_lrs' in self.params \
                            and 'manual_decay_iters' in self.params:
                        decay_index = 0
                        iter_counter = 0
                        for decay_i, manual_decay_iter in enumerate(self.params['manual_decay_iters']):
                            iter_counter += manual_decay_iter
                            if decay_iters <= iter_counter:
                                decay_index = decay_i
                                break
                        learning_rate = self.params['manual_decay_lrs'][decay_index]
                    else:
                        decay_interval = self.params['decay_interval']
                        decay_after = self.params.get('decay_after', 0)
                        if decay_iters >= decay_after:
                            decay_minimum = decay_after / decay_interval
                            decay = (decay_iters // decay_interval) - decay_minimum
                            learning_rate = self.params['learning_rate'] / (2 ** decay)
                        else:
                            learning_rate = self.params['learning_rate']

                    feed_dict = {learning_rate_: learning_rate, global_step_: i}
                    try:
                        _, loss = sess.run(
                            [train_op, loss_],
                            feed_dict=feed_dict,
                            options=run_options,
                            run_metadata=run_metadata)

                        # Save last successfull index
                        save_iter = max(i, save_iter)

                        if i == 1 or i % self.params['display_interval'] == 0:
                            summary = sess.run(summary_, feed_dict=feed_dict)
                            summary_writer.add_summary(summary, i)
                            print("-- train: i = {}, loss = {}".format(i, loss))

                    except tf.errors.OutOfRangeError as exc:
                        print(exc)
                        print("error in iteration {}".format(i))
                        coord.request_stop()

                save_path = os.path.join(self.ckpt_dir, 'model.ckpt')
                print("save iteration {}".format(save_iter))
                saver.save(sess, save_path, global_step=save_iter)

                summary_writer.close()
                coord.request_stop()
                try:
                    coord.join(threads)
                except tf.errors.InvalidArgumentError as exc:
                    print(exc)
                    # Make sure all threads are dead
                    if any(t.is_alive() for t in threads):
                        raise Exception("Not all threads are dead...")
                    print("All threads are dead, continue!")

                return save_iter

    def eval(self, num):
        assert num == 1  # TODO enable num > 1

        with tf.Graph().as_default():
            inputs = self.eval_batch_fn()
            im1, im2, input_shape, intrinsics = inputs[:4]
            truths = inputs[4:]  # empty if dataset has no groutndtruth flow
            height, width, _ = tf.unstack(tf.squeeze(input_shape), num=3, axis=0)
            # im1 = resize_input(im1, height, width, 384, 1280)
            # im2 = resize_input(im2, height, width, 384, 1280)

            _, flow, flow_bw, motion_angles, inlier_prob_mask = unsupervised_loss(
                (im1, im2, input_shape, intrinsics),
                params=self.params,
                normalization=self.normalization,
                augment_photometric=False, return_flow=True)

            # im1 = resize_output(im1, height, width, 3)
            # im2 = resize_output(im2, height, width, 3)
            # flow = resize_output_flow(flow, height, width, 2)
            # flow_bw = resize_output_flow(flow_bw, height, width, 2)

            variables_to_restore = tf.all_variables()

            images_ = [image_warp(im1, flow) / 255,
                       flow_to_color(flow),
                       1 - (1 - occlusion(flow, flow_bw)[0]) * create_outgoing_mask(flow),
                       forward_warp(flow_bw) < DISOCC_THRESH]
            image_names = ['warped image', 'flow', 'occ', 'reverse disocc']

            values_ = []
            averages_ = []
            truth_tuples = []
            if len(truths) == 4:
                flow_occ, mask_occ, flow_noc, mask_noc = truths
                flow_occ = resize_output_crop(flow_occ, height, width, 2)
                flow_noc = resize_output_crop(flow_noc, height, width, 2)
                mask_occ = resize_output_crop(mask_occ, height, width, 1)
                mask_noc = resize_output_crop(mask_noc, height, width, 1)

                truth_tuples.append(('occluded', flow_occ, mask_occ))
                truth_tuples.append(('non-occluded', flow_noc, mask_noc))
                images_ += [flow_error_image(flow, flow_occ, mask_occ, mask_noc)]
                image_names += ['flow error']
            elif len(truths) == 2:
                raise NotImplementedError
                truth_tuples.append(('flow', truths[0], truths[1]))

            for name, gt_flow, mask in truth_tuples:
                error_ = flow_error_avg(gt_flow, flow, mask)
                error_avg_ = summarized_placeholder('AEE/' + name, key='eval_avg')
                outliers_ = outlier_pct(gt_flow, flow, mask)
                outliers_avg = summarized_placeholder('outliers/' + name,
                                                      key='eval_avg')
                values_.extend([error_, outliers_])
                averages_.extend([error_avg_, outliers_avg])

            losses = tf.get_collection('losses')
            for l in losses:
                values_.append(l)
                tensor_name = re.sub('tower_[0-9]*/', '', l.op.name)
                loss_avg_ = summarized_placeholder(tensor_name, key='eval_avg')
                averages_.append(loss_avg_)

            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            assert ckpt is not None, "No checkpoints to evaluate"

            # Correct path for ckpts from different machine
            # ckpt_path = self.ckpt_dir + "/" + os.path.basename(ckpt.model_checkpoint_path)
            ckpt_path = ckpt.model_checkpoint_path

            with tf.Session() as sess:
                summary_writer = tf.summary.FileWriter(self.eval_summaries_dir)
                saver = tf.train.Saver(variables_to_restore)

                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                restore_networks(sess, self.params, ckpt)
                global_step = ckpt_path.split('/')[-1].split('-')[-1]

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess,
                                                       coord=coord)
                averages = np.zeros(len(averages_))
                num_iters = 0

                image_lists = []
                try:
                    while not coord.should_stop():
                        results = sess.run(values_ + images_)
                        values = results[:len(averages_)]
                        images = results[len(averages_):]
                        image_lists.append(images)
                        averages += values
                        num_iters += 1
                except tf.errors.OutOfRangeError:
                    pass

                averages /= num_iters
                feed = {k: v for (k, v) in zip(averages_, averages)}

                summary_ = tf.summary.merge_all('eval_avg')
                summary = sess.run(summary_, feed_dict=feed)
                summary_writer.add_summary(summary, global_step)

                print("-- eval: i = {}".format(global_step))

                coord.request_stop()
                coord.join(threads)
                summary_writer.close()

                if self.interactive_plot:
                    if self.plot_proc:
                        self.plot_proc.terminate()
                    self.plot_proc = Process(target=_eval_plot,
                                             args=([image_lists], image_names,
                                                   "{} (i={})".format(self.experiment,
                                                                      global_step)))
                    self.plot_proc.start()


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            if g is not None:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
        if grads != []:
            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads
