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

    return make_intrinsics_matrix(fx, fy, cx, cy)


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

    def _preprocess_image(self, image, calib_tf=None):
        scale = 1300.0 / 2048.
        h, w, c = tf.unstack(tf.shape(image))
        image = tf.image.resize_bilinear(image, [h * scale, w * scale])
        if calib_tf:
            calib_tf = scale_intrinsics(calib_tf, scale, scale)
        return image, calib_tf
