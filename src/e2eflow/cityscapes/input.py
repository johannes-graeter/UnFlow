import json

import numpy as np
import tensorflow as tf

from ..core.input import Input
from ..core.augment import make_intrinsics_matrix


def scale_intrinsics(calib, sx, sy):
    fx = tf.expand_dims(calib[0, 0] * sx, axis=0)
    cx = tf.expand_dims(calib[0, 2] * sx, axis=0)
    fy = tf.expand_dims(calib[1, 1] * sy, axis=0)
    cy = tf.expand_dims(calib[1, 2] * sy, axis=0)

    return tf.squeeze(make_intrinsics_matrix(fx, fy, cx, cy), axis=0)


class CityscapesInput(Input):
    def __init__(self, data, batch_size, dims=(320, 1152), *,
                 num_threads=1, normalize=True, skipped_frames=False
                 ):
        super().__init__(data, batch_size, dims, num_threads=num_threads,
                         normalize=normalize, skipped_frames=skipped_frames)

    def _decode_calib(self, string_tensor, key):
        def get_intrin_np(s):
            obj = json.loads(s.decode('utf-8'))
            d = obj['intrinsic']
            return np.array([[d['fx'], 0., d['u0']], [0., d['fy'], d['v0']], [0., 0., 1.]], dtype=np.float32)

        [parsed] = tf.py_func(get_intrin_np, [string_tensor], [tf.float32])
        # parsed = tf.reshape(parsed,[3, 3])  # fix the size
        parsed.set_shape([3, 3])  # fix the size

        return parsed

    def _frame_name_to_num(self, name):
        stripped = name.split("/")[-1].split(".")[-2].split("_")[-2]
        if stripped == '':
            return 0
        return int(stripped)

    def _preprocess_image(self, image, calib_tf=None):
        scale = 1200.0 / 2048.
        crop_bottom = 200
        h, w, c = tf.unstack(tf.shape(image))
        image = tf.expand_dims(image, axis=0)
        tf.image.crop_to_bounding_box(image, 0, 0, h - crop_bottom, w)
        image = tf.image.resize_bilinear(image,
                                         [tf.cast(tf.scalar_mul(scale, tf.cast(h - crop_bottom, tf.float32)), tf.int32),
                                          tf.cast(tf.scalar_mul(scale, tf.cast(w, tf.float32)), tf.int32)])
        image = tf.squeeze(image, axis=0)
        if calib_tf is not None:
            calib_tf = scale_intrinsics(calib_tf, scale, scale)
        return image, calib_tf
