
import torch.utils.data as data

import os
import cv2
import numpy as np
from data import common


class SRData(data.Dataset):
    def __init__(self, args, root_dir, train=True):
        self.args = args
        self.train = train
        self.scale = args.scale[0]
        self.root_dir = root_dir
        self._set_filesystem()

        self.hr_img_dirs, self.lr_img_dirs = self._scan()
        if args.ext == 'img':
            print('Read data from image files')
        elif args.ext.find('sep') >= 0:
            print('Read data from binary files')
            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                for v in self.hr_img_dirs:
                    hr = cv2.imread(v, cv2.IMREAD_COLOR)
                    hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, hr)
                for v in self.lr_img_dirs:
                    lr = cv2.imread(v, cv2.IMREAD_COLOR)
                    lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, lr)

            self.hr_img_dirs = [
                v.replace(self.ext, '.npy') for v in self.hr_img_dirs
            ]
            self.lr_img_dirs = [
                v.replace(self.ext, '.npy') for v in self.lr_img_dirs
            ]
        elif args.ext == 'ram':
            print('Read data from RAM')
            hr_img_size = self.args.image_size
            lr_img_size = int(self.args.image_size/self.scale)
            img_nums = len(self.hr_img_dirs)
            self.hr_img_files = np.zeros((img_nums, hr_img_size, hr_img_size, self.args.n_colors), dtype=np.uint8)
            self.lr_img_files = np.zeros((img_nums, lr_img_size, lr_img_size, self.args.n_colors), dtype=np.uint8)
            for i, v in enumerate(self.hr_img_dirs):
                hr = cv2.imread(v, cv2.IMREAD_COLOR)
                hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
                # self.hr_img_files.append(hr)
                self.hr_img_files[i, :, :, :] = hr
            for i, v in enumerate(self.lr_img_dirs):
                lr = cv2.imread(v, cv2.IMREAD_COLOR)
                lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
                # self.lr_img_files.append(lr)
                self.lr_img_files[i, :, :, :] = lr
        else:
            print('Please define data type')

    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        if self.train or self.args.test_patch:
            lr, hr = self._get_patch(lr, hr)
        if self.args.cubic_input:
            lr = cv2.resize(lr, (hr.shape[0], hr.shape[1]), interpolation=cv2.INTER_CUBIC)
        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
        return lr_tensor, hr_tensor, filename

    def _load_file(self, idx):
        """ load lr and hr image files"""
        lr_dir = self.lr_img_dirs[idx]
        hr_dir = self.hr_img_dirs[idx]
        filename = hr_dir
        if self.args.ext == 'img':
            lr = cv2.imread(lr_dir, cv2.IMREAD_COLOR)
            lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
            hr = cv2.imread(hr_dir, cv2.IMREAD_COLOR)
            hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        elif self.args.ext.find('sep') >= 0:
            lr = np.load(lr_dir)
            hr = np.load(hr_dir)
        elif self.args.ext == 'ram':
            lr = self.lr_img_files[idx, :, :, :]
            hr = self.hr_img_files[idx, :, :, :]
        filename = os.path.splitext(os.path.split(filename)[-1])[0]
        return lr, hr, filename

    def _get_patch(self, lr, hr):
        """ get patch from lr and hr images"""
        lr, hr = common.get_patch(
            lr, hr, self.args.patch_size, self.scale)
        if self.train:
            lr, hr = common.augment([lr, hr])
            lr = common.add_noise(lr, self.args.noise)
        return lr, hr




