import glob
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

    def _get_paths(self, folder_name):
        split = "train"
        img_dir = os.path.join(self.current_dir, folder_name, split)
        city_list = os.listdir(img_dir)
        dirs = []
        for city in city_list:
            p = os.path.join(img_dir, city)
            # Get paths should be sth. like ../../train/aachen/aachen_000212
            city_paths = [os.join(p, "_".join(n.split("/")[-1].split("_")[:2])) for n in
                          glob.glob(os.join(p, city + "_*"))]
            # Treat every glob as own path.
            dirs.extend(city_paths)
        return dirs

    def get_raw_dirs(self):
        dirs = []
        for extract_path in self._get_paths('leftImg8bit_sequence'):
            image_folder = [os.path.join(extract_path, n) for n in self.image_subdirs]
            dirs.extend(image_folder)
        return dirs

    def get_intrinsic_dirs(self):
        calibs = {}
        for path in zip(self._get_paths('camera')):
            calib_file = glob.glob(os.path.join(path, "*.json"))
            assert (len(calib_file) == 1)
            calib_file = calib_file[0]
            calibs[path] = [calib_file, ""]
        return calibs
