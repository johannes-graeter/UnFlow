import os

from . import raw_records
from ..core.data import Data
from ..util import tryremove
from .input import frame_name_to_num_kitti


def exclude_test_and_train_images(kitti_dir, exclude_lists_dir, exclude_target_dir,
                                  remove=False):
    to_move = []

    def exclude_from_seq(day_name, seq_str, image, view, distance=10):
        # image is the first frame of each frame pair to exclude
        seq_dir_rel = os.path.join(day_name, seq_str, view, 'data')
        seq_dir_abs = os.path.join(kitti_dir, seq_dir_rel)
        target_dir_abs = os.path.join(exclude_target_dir, seq_dir_rel)
        if not os.path.isdir(seq_dir_abs):
            print("Not found: {}".format(seq_dir_abs))
            return
        try:
            os.makedirs(target_dir_abs)
        except:
            pass
        seq_files = sorted(os.listdir(seq_dir_abs))
        image_num = frame_name_to_num_kitti(image)
        try:
            image_index = seq_files.index(image)
        except ValueError:
            return
        # assume that some in-between files may be missing
        start = max(0, image_index - distance)
        stop = min(len(seq_files), image_index + distance + 2)
        start_num = image_num - distance
        stop_num = image_num + distance + 2
        for i in range(start, stop):
            filename = seq_files[i]
            num = frame_name_to_num_kitti(filename)
            if num < start_num or num >= stop_num:
                continue
            to_move.append((os.path.join(seq_dir_abs, filename),
                            os.path.join(target_dir_abs, filename)))

    for filename in os.listdir(exclude_lists_dir):
        exclude_list_path = os.path.join(exclude_lists_dir, filename)
        with open(exclude_list_path) as f:
            for line in f:
                line = line.rstrip('\n')
                if line.split(' ')[0].endswith('_10'):
                    splits = line.split(' ')[-1].split('\\')
                    image = splits[-1]
                    seq_str = splits[0]
                    day_name, seq_name = seq_str.split('_drive_')
                    seq_name = seq_name.split('_')[0] + '_extract'
                    seq_str = day_name + '_drive_' + seq_name
                    exclude_from_seq(day_name, seq_str, image, 'image_02')
                    exclude_from_seq(day_name, seq_str, image, 'image_03')
    if remove:
        print("Collected {} files. Deleting...".format(len(to_move)))
    else:
        print("Collected {} files. Moving...".format(len(to_move)))

    for i, data in enumerate(to_move):
        try:
            src, dst = data
            print("{} / {}: {}".format(i, len(to_move) - 1, src))
            if remove:
                os.remove(src)
            else:
                os.rename(src, dst)
        except:  # Some ranges may overlap
            pass

    return len(to_move)


def get_dir(path):
    return [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]


class KITTIDataRaw(Data):
    def __init__(self, data_dir, stat_log_dir=None, do_fetch=True,
                 development=True, fast_dir=None):
        super().__init__(data_dir, stat_log_dir, do_fetch=do_fetch,
                         development=development,
                         fast_dir=fast_dir)
        self.url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'
        self.dir = 'kitti_raw'
        self.image_subdirs = ['image_02/data', 'image_03/data']
        self.calib_identifiers = ['P_rect_02', 'P_rect_03']
        self.calib_name = 'calib_cam_to_cam.txt'

    def _fetch_if_missing(self):
        self._maybe_get_data()

    def _get_paths(self):
        """Get paths for extracted data and calibs.
        Sometimes that needs to be overloaded because there could be one calib for
        one sequence (odometry) or many sequences (kitti_raw)"""
        top_dir = os.path.join(self.current_dir, self.dir)
        dates = get_dir(top_dir)
        extract_paths = []
        calib_paths = []
        for date in dates:
            date_path = os.path.join(top_dir, date)
            extracts = get_dir(date_path)
            for extract in extracts:
                extract_paths.append(os.path.join(date_path, extract))
                calib_paths.append(date_path)
        return extract_paths, calib_paths

    def get_raw_dirs(self):
        dirs = []
        for extract_path in self._get_paths()[0]:
            image_folder = [os.path.join(extract_path, n) for n in self.image_subdirs]
            dirs.extend(image_folder)
        return dirs

    def get_intrinsic_dirs(self):
        calibs = {}
        for extract_path, calib_path in zip(*self._get_paths()):
            image_folders = [os.path.join(extract_path, n) for n in self.image_subdirs]
            calib_file = os.path.join(calib_path, self.calib_name)
            for f, ident in zip(image_folders, self.calib_identifiers):
                calibs[f] = [calib_file, ident]
        return calibs

    def _maybe_get_data(self):
        base_url = self.url
        local_dir = os.path.join(self.data_dir, self.dir)
        records = raw_records.get_kitti_records(self.development)
        downloaded_records = False

        for i, record in enumerate(records):
            date_str = record.split("_drive_")[0]
            foldername = record + "_extract"
            date_folder = os.path.join(local_dir, date_str)
            if not os.path.isdir(date_folder):
                os.makedirs(date_folder)
            local_path = os.path.join(date_folder, foldername)
            if not os.path.isdir(local_path):
                url = base_url + record + "/" + foldername + '.zip'
                print(url)
                self._download_and_extract(url, local_dir)
                downloaded_records = True

            # Remove unused directories
            tryremove(os.path.join(local_path, 'velodyne_points'))
            tryremove(os.path.join(local_path, 'oxts'))
            tryremove(os.path.join(local_path, 'image_00'))
            tryremove(os.path.join(local_path, 'image_01'))

        if downloaded_records:
            print("Downloaded all KITTI raw files.")
            exclude_target_dir = os.path.join(self.data_dir, 'exclude_target_dir')
            exclude_lists_dir = '../files/kitti_excludes'
            excluded = exclude_test_and_train_images(local_dir, exclude_lists_dir, exclude_target_dir,
                                                     remove=True)


class KITTIDataOdometry(KITTIDataRaw):
    def __init__(self, data_dir, stat_log_dir=None, do_fetch=True,
                 development=True, fast_dir=None):
        super().__init__(data_dir, stat_log_dir, do_fetch=do_fetch,
                         development=development,
                         fast_dir=fast_dir)
        # KITTI_2015_URL = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip'
        self.dir = './'
        self.image_subdirs = ['image_2', 'image_3']
        self.calib_identifiers = ['P2', 'P3']
        self.calib_name = 'calib.txt'

    def _fetch_if_missing(self):
        raise Exception("Can not download odometry automatically get it manually.")

    def _get_paths(self):
        """Odometry benchmark has one hierarchy level less than raw dataset"""
        top_dir = os.path.join(self.current_dir, self.dir)
        sequs = get_dir(top_dir)
        sequ_path = [os.path.join(top_dir, sequ) for sequ in sequs]

        # sequ paths and calib paths are identical for this dataset
        return sequ_path, sequ_path

    def _maybe_get_data(self):
        pass

# class KITTIDataFlow(KITTIDataRaw):
#     def __init__(self, data_dir, stat_log_dir=None, do_fetch=True,
#                  development=True, fast_dir=None):
#         super().__init__(data_dir, stat_log_dir, do_fetch=do_fetch,
#                          development=development,
#                          fast_dir=fast_dir)
#         self.url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip'
#         # KITTI_2015_URL = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip'
#         self.dir = 'data_stereo_flow'
#         self.image_subdirs = ['image_02/data', 'image_03/data']
#         self.calib_identifiers = ['P_rect_02', 'P_rect_03']
#
#     def _fetch_if_missing(self):
#         self._maybe_get_kitti_2012()
#         self._maybe_get_kitti_2015()
#
#     def _maybe_get_kitti_2012(self):
#         local_path = os.path.join(self.data_dir, 'data_stereo_flow')
#         if not os.path.isdir(local_path):
#             self._download_and_extract(self.url, local_path)

#    def _maybe_get_kitti_2015(self):
#        local_path = os.path.join(self.data_dir, 'data_scene_flow')
#        if not os.path.isdir(local_path):
#            self._download_and_extract(self.KITTI_2015_URL, local_path)
