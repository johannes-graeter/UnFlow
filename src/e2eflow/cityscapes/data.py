import os

from ..core.data import Data


class CityscapesData(Data):
    dirs = ['cs']

    def __init__(self, data_dir, stat_log_dir=None,
                 development=True, fast_dir=None):
        super().__init__(data_dir, stat_log_dir,
                         development=development,
                         fast_dir=fast_dir)

    def _fetch_if_missing(self):
        pass

    def get_raw_dirs(self):
        split = "train"
        img_dir = os.path.join(self.current_dir, 'leftImg8bit_sequence', split)
        city_list = os.listdir(img_dir)
        dirs = []
        for city in city_list:
            city_path = os.path.join(img_dir, city)
            dirs.append(city_path)
        return dirs

    def get_intrinsic_dirs(self):
        calibs = {}
        for extract_path, calib_path in zip(*self._get_paths()):
            image_folders = [os.path.join(extract_path, n) for n in self.image_subdirs]
            calib_file = os.path.join(calib_path, self.calib_name)
            for f, ident in zip(image_folders, self.calib_identifiers):
                calibs[f] = [calib_file, ident]
        return calibs