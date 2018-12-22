import glob
import os

from ..core.data import Data


class CityscapesData(Data):
    def __init__(self, data_dir, sub_dir="train", stat_log_dir=None,
                 development=True, fast_dir=None):
        super().__init__(data_dir, stat_log_dir,
                         development=development,
                         fast_dir=fast_dir,
                         do_fetch=False
                         )
        self.sub_dir = sub_dir

    def _fetch_if_missing(self):
        raise NotImplementedError("Fetching for cityscapes data not implemented. Download it manually.")

    def _get_paths(self, folder_name):
        img_dir = os.path.join(self.current_dir, folder_name, self.sub_dir)
        city_list = os.listdir(img_dir)
        dirs = []
        for city in sorted(city_list):
            p = os.path.join(img_dir, city)
            # Get paths should be sth. like ../../train/aachen/aachen_000212
            city_paths = set([os.path.join(p, "_".join(n.split("/")[-1].split("_")[:2])) for n in
                              glob.glob(os.path.join(p, city + "_*"))])
            # Treat every glob as own path.
            dirs.extend(sorted(city_paths))
        return dirs

    def get_raw_dirs(self):
        return self._get_paths('leftImg8bit_sequence')

    def get_intrinsic_dirs(self):
        calibs = {}
        c_paths = self._get_paths('camera')
        sequ_paths = self.get_raw_dirs()
        assert (len(c_paths) == len(sequ_paths))
        for c_path, sequ_path in zip(c_paths, sequ_paths):
            calib_file = glob.glob(c_path + "*.json")
            assert (len(calib_file) > 0)
            calib_file = calib_file[0]
            calibs[sequ_path] = [calib_file, ""]
        return calibs
