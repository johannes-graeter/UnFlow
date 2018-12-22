import glob
import os
import random

import numpy as np
import tensorflow as tf

from .augment import data_augmentation


def resize_input(t, height, width, resized_h, resized_w):
    # Undo old resizing and apply bilinear
    t = tf.reshape(t, [resized_h, resized_w, 3])
    t = tf.expand_dims(tf.image.resize_image_with_crop_or_pad(t, height, width), 0)
    return tf.image.resize_bilinear(t, [resized_h, resized_w])


def resize_output_crop(t, height, width, channels):
    _, oldh, oldw, c = tf.unstack(tf.shape(t))
    t = tf.reshape(t, [oldh, oldw, c])
    t = tf.image.resize_image_with_crop_or_pad(t, height, width)
    return tf.reshape(t, [1, height, width, channels])


def resize_output(t, height, width, channels):
    return tf.image.resize_bilinear(t, [height, width])


def resize_output_flow(t, height, width, channels):
    batch, old_height, old_width, _ = tf.unstack(tf.shape(t), num=4)
    t = tf.image.resize_bilinear(t, [height, width])
    u, v = tf.unstack(t, axis=3)
    u *= tf.cast(width, tf.float32) / tf.cast(old_width, tf.float32)
    v *= tf.cast(height, tf.float32) / tf.cast(old_height, tf.float32)
    return tf.reshape(tf.stack([u, v], axis=3), [batch, height, width, 2])


def frame_name_to_num(name):
    stripped = name.split('.')[0].lstrip('0')
    if stripped == '':
        return 0
    return int(stripped)


class Input:
    mean = [104.920005, 110.1753, 114.785955]
    stddev = 1 / 0.0039216

    def __init__(self, data, batch_size, dims, *,
                 num_threads=1, normalize=True,
                 skipped_frames=False):
        assert len(dims) == 2
        self.data = data
        self.dims = dims
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.normalize = normalize
        self.skipped_frames = skipped_frames

    def _resize_crop_or_pad(self, tensor, calib=None):
        height, width = self.dims

        if calib is not None:
            orig_shape = tf.shape(tensor)
            # orig_shape = tf.Print(orig_shape, ["orig_shape", orig_shape])
            dh = tf.scalar_mul(0.5, tf.cast(orig_shape[0] - height, dtype=tf.float32))
            dw = tf.scalar_mul(0.5, tf.cast(orig_shape[1] - width, dtype=tf.float32))
            # dh = tf.Print(dh, ["height diff", dh])
            # dw = tf.Print(dw, ["width diff", dw])

            corr = tf.concat((tf.zeros((3, 2)), [[dw], [dh], [0.]]), axis=1)
            calib = calib - corr

        return tf.image.resize_image_with_crop_or_pad(tensor, height, width), calib

    def _resize_image_fixed(self, image, calib=None):
        height, width = self.dims
        return tf.reshape(self._resize_crop_or_pad(image, calib), [height, width, 3])

    def _normalize_image(self, image):
        return (image - self.mean) / self.stddev

    def _preprocess_image(self, image):
        image, _ = self._resize_image_fixed(image)
        if self.normalize:
            image = self._normalize_image(image)
        return image

    def _input_images(self, image_dir, hold_out_inv=None):
        """Assumes that paired images are next to each other after ordering the
        files.
        """
        image_dir = os.path.join(self.data.current_dir, image_dir)

        filenames_1 = []
        filenames_2 = []
        image_files = os.listdir(image_dir)
        image_files.sort()

        assert len(image_files) % 2 == 0, 'expected pairs of images'

        for i in range(len(image_files) // 2):
            filenames_1.append(os.path.join(image_dir, image_files[i * 2]))
            filenames_2.append(os.path.join(image_dir, image_files[i * 2 + 1]))

        if hold_out_inv is not None:
            filenames = list(zip(filenames_1, filenames_2))
            random.seed(0)
            random.shuffle(filenames)
            filenames = filenames[:hold_out_inv]

            filenames_1, filenames_2 = zip(*filenames)
            filenames_1 = list(filenames_1)
            filenames_2 = list(filenames_2)

        input_1 = read_png_image(filenames_1, 1)
        input_2 = read_png_image(filenames_2, 1)
        image_1 = self._preprocess_image(input_1)
        image_2 = self._preprocess_image(input_2)

        image_1 = tf.print(image_1, [image_1.shape.as_list()])
        return tf.shape(input_1), image_1, image_2

    def _input_test(self, image_dir, hold_out_inv=None):
        input_shape, im1, im2 = self._input_images(image_dir, hold_out_inv)
        return tf.train.batch(
            [im1, im2, input_shape],
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            allow_smaller_final_batch=True)

    def get_normalization(self):
        return self.mean, self.stddev

    def _decode_calib(self, string_tensor, key):
        raise NotImplementedError("core::input::_decode_calib is not implemented")

    def read_calib(self, filenames, keys):
        """Given a 2 lists of filenames and keys, constructs a reader op for calibs."""
        filename_queue = tf.train.string_input_producer(filenames, shuffle=False, capacity=len(filenames))
        key_queue = tf.train.string_input_producer(keys, shuffle=False, capacity=len(keys))
        reader = tf.WholeFileReader()
        _, value = reader.read(filename_queue)

        # decode
        key = key_queue.dequeue()  # does that really dequeue at the same time as filenames?
        calib = self._decode_calib(value, key)

        return calib

    def input_raw(self, swap_images=True, sequence=True,
                  augment_crop=True, shift=0, seed=0,
                  center_crop=False, skip=0):
        """Constructs input of raw data.

        Args:
            sequence: Assumes that image file order in data_dirs corresponds to
                temporal order, if True. Otherwise, assumes uncorrelated pairs of
                images in lexicographical ordering.
            shift: number of examples to shift the input queue by.
                Useful to resume training.
            swap_images: for each pair (im1, im2), also include (im2, im1)
            seed: seed for filename shuffling.
        Returns:
            image_1: batch of first images
            image_2: batch of second images
        """
        if not isinstance(skip, list):
            skip = [skip]

        data_dirs = self.data.get_raw_dirs()
        intrinsic_dirs = self.data.get_intrinsic_dirs()
        height, width = self.dims

        filenames = []
        for dir_path in data_dirs:

            files = glob.glob(dir_path + "*.png")  # That also support cityscapes.
            files.sort()
            if sequence:
                steps = [1 + s for s in skip]
                stops = [len(files) - s for s in steps]
            else:
                steps = [2]
                stops = [len(files)]
                assert len(files) % 2 == 0
            for step, stop in zip(steps, stops):
                for i in range(0, stop, step):
                    if self.skipped_frames and sequence:
                        assert step == 1
                        num_first = frame_name_to_num(files[i])
                        num_second = frame_name_to_num(files[i + 1])
                        if num_first + 1 != num_second:
                            continue
                    fn1 = os.path.join(dir_path, files[i])
                    fn2 = os.path.join(dir_path, files[i + 1])
                    calib_dir, key = intrinsic_dirs[dir_path]
                    filenames.append((fn1, fn2, calib_dir, key))

        if seed:
            random.seed(seed)
            random.shuffle(filenames)
        print("Input {} frame pairs.".format(len(filenames)))

        filenames_extended = []
        # print("fs", filenames[0])
        for fn1, fn2, calib, key in filenames:
            filenames_extended.append((fn1, fn2, calib, key))
            if swap_images:
                filenames_extended.append((fn2, fn1, calib, key))

        shift = shift % len(filenames_extended)
        filenames_extended = list(np.roll(filenames_extended, shift, axis=0))

        filenames_1, filenames_2, calib_dir, key = zip(*filenames_extended)
        filenames_1 = list(filenames_1)
        filenames_2 = list(filenames_2)

        # unpack calid files and keys
        calib_filenames = list(calib_dir)
        keys = list(key)

        with tf.variable_scope('train_inputs'):
            image_1 = read_png_image(filenames_1)
            image_2 = read_png_image(filenames_2)
            calib_tf = self.read_calib(calib_filenames, keys)

            shape_before_preproc = tf.shape(image_1)

            if augment_crop:
                out_height, out_width = self.dims
                #img_h, img_w, ch = image_1.shape.as_list()
                #if (out_height > img_h) or (out_width > img_w):
                    #raise Exception("No crop for augmentation possible, input image too small.")
                image_1, image_2, calib_tf = data_augmentation(image_1, image_2, calib_tf, out_h=out_height,
                                                               out_w=out_width)
            elif center_crop:
                image_1, calib_tf = self._resize_crop_or_pad(image_1, calib_tf)
                image_2, _ = self._resize_crop_or_pad(image_2)
            else:
                image_1 = tf.reshape(image_1, [height, width, 3])
                image_2 = tf.reshape(image_2, [height, width, 3])

            if self.normalize:
                image_1 = self._normalize_image(image_1)
                image_2 = self._normalize_image(image_2)
            print(calib_tf)

            return tf.train.batch(
                [image_1, image_2, shape_before_preproc, calib_tf],
                batch_size=self.batch_size,
                num_threads=self.num_threads)


def read_png_image(filenames):
    """Given a list of filenames, constructs a reader op for images."""
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False, capacity=len(filenames))
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_uint8 = tf.image.decode_png(value, channels=3)
    image = tf.cast(image_uint8, tf.float32)
    return image
