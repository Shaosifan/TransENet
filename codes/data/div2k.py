

import os
import glob
from data import srdata


class DIV2KDataset(srdata.SRData):
    def __init__(self, args, root_dir, train=True):
        super(DIV2KDataset, self).__init__(args, root_dir, train)
        self._set_filesystem()
        self.train = train

    def _set_filesystem(self):
        self.dir_hr = os.path.join(self.root_dir, 'HR')
        self.dir_lr = os.path.join(self.root_dir, 'LR_x4')
        self.ext = '.png'

    def _scan(self):
        list_hr = glob.glob(os.path.join(self.dir_hr, '*'+self.ext))
        list_lr = []
        for i in range(len(list_hr)):
            filename = os.path.split(list_hr[i])[-1]
            list_lr.append(os.path.join(self.dir_lr, filename))

        return list_hr, list_lr

    def __len__(self):
        return len(self.hr_img_dirs)














