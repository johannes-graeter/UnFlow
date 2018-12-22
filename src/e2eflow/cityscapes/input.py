import json

import numpy as np
import tensorflow as tf

from ..core.input import Input


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
        #parsed = tf.reshape(parsed,[3, 3])  # fix the size
        parsed.set_shape([3, 3])  # fix the size

        return parsed
