import json
import os

import numpy as np
import tensorflow as tf

from ..core.input import Input


class CityscapesInput(Input):
    """Inspired by SFMLearner"""

    def __init__(self, data, batch_size, dims=(320, 1152), *,
                 num_threads=1, normalize=True,
                 crop_bottom=True,  # Get rid of the car logo
                 sample_gap=2,  # Sample every two frames to match KITTI frame rate
                 skipped_frames=False,
                 seq_length=5
                 ):
        super().__init__(data, batch_size, dims, num_threads=num_threads,
                         normalize=normalize, skipped_frames=skipped_frames)

        # Crop out the bottom 25% of the image to remove the car logo
        self.crop_bottom = crop_bottom
        self.sample_gap = sample_gap
        self.img_height = dims[0]
        self.img_width = dims[1]
        self.seq_length = seq_length
        assert seq_length % 2 != 0, 'seq_length must be odd!'

    def _decode_calib(self, string_tensor, key):
        def get_intrin_np(s):
            obj = json.loads(s.decode('utf-8'))
            d = obj['intrinsic']
            return np.array([[d['fx'], 0., d['u0']], [0., d['fy'], d['v0']], [0., 0., 1.]], dtype=np.float32)

        [parsed] = tf.py_func(get_intrin_np, [string_tensor], [tf.float32])

        return parsed

    def get_train_example_with_idx(self, tgt_idx):
        tgt_frame_id = self.frames[tgt_idx]
        if not self.is_valid_example(tgt_frame_id):
            return False
        example = self.load_example(self.frames[tgt_idx])
        return example

    def is_valid_example(self, tgt_frame_id):
        city, snippet_id, tgt_local_frame_id, _ = tgt_frame_id.split('_')
        half_offset = int((self.seq_length - 1) / 2 * self.sample_gap)
        for o in range(-half_offset, half_offset + 1, self.sample_gap):
            curr_local_frame_id = '%.6d' % (int(tgt_local_frame_id) + o)
            curr_frame_id = '%s_%s_%s_' % (city, snippet_id, curr_local_frame_id)
            curr_image_file = os.path.join(self.dataset_dir, 'leftImg8bit_sequence',
                                           self.split, city, curr_frame_id + 'leftImg8bit.png')
            if not os.path.exists(curr_image_file):
                return False
        return True

    def load_image_sequence(self, tgt_frame_id, seq_length, crop_bottom):
        city, snippet_id, tgt_local_frame_id, _ = tgt_frame_id.split('_')
        half_offset = int((self.seq_length - 1) / 2 * self.sample_gap)
        image_seq = []
        for o in range(-half_offset, half_offset + 1, self.sample_gap):
            curr_local_frame_id = '%.6d' % (int(tgt_local_frame_id) + o)
            curr_frame_id = '%s_%s_%s_' % (city, snippet_id, curr_local_frame_id)
            curr_image_file = os.path.join(self.dataset_dir, 'leftImg8bit_sequence',
                                           self.split, city, curr_frame_id + 'leftImg8bit.png')
            curr_img = scipy.misc.imread(curr_image_file)
            raw_shape = np.copy(curr_img.shape)
            if o == 0:
                zoom_y = self.img_height / raw_shape[0]
                zoom_x = self.img_width / raw_shape[1]
            curr_img = scipy.misc.imresize(curr_img, (self.img_height, self.img_width))
            if crop_bottom:
                ymax = int(curr_img.shape[0] * 0.75)
                curr_img = curr_img[:ymax]
            image_seq.append(curr_img)
        return image_seq, zoom_x, zoom_y

    def load_example(self, tgt_frame_id, load_gt_pose=False):
        image_seq, zoom_x, zoom_y = self.load_image_sequence(tgt_frame_id, self.seq_length, self.crop_bottom)
        intrinsics = self.load_intrinsics(tgt_frame_id, self.split)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = tgt_frame_id.split('_')[0]
        example['file_name'] = tgt_frame_id[:-1]
        return example
