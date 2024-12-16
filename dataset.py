"""
Dataset related functions

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import os.path
import random
import glob
import numpy as np
import cv2
import h5py
import torch
import torch.utils.data as udata
from PIL import Image

from utils import data_augmentation, normalize
class Dataset(udata.Dataset):
    r"""Implements torch.utils.data.Dataset
    """

    def __init__(self, train=True, gray_mode=False, shuffle=False):
        super(Dataset, self).__init__()
        self.train = train
        self.gray_mode = gray_mode
        if not self.gray_mode:
            self.traindbf = 'train_rgb.h5'
            self.valdbf = 'val_rgb.h5'
            self.valdirtydbf = 'val_dirty_rgb.h5'
        else:
            self.traindbf = 'train_gray.h5'
            self.valdbf = 'val_gray.h5'
            self.valdirtydbf = 'val_dirty_gray.h5'

        if self.train:
            h5f = h5py.File(self.traindbf, 'r')
            self.keys = list(h5f.keys())
            if shuffle:
                random.shuffle(self.keys)
            h5f.close()
        else:
            h5f = h5py.File(self.valdbf, 'r')
            h5f_dirty = h5py.File(self.valdirtydbf, 'r')
            self.keys = list(h5f.keys())
            if shuffle:
                random.shuffle(self.keys)
            h5f.close()
            h5f_dirty.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # 从 计算机的具体路径下 读图片 转化为 pytroch框架可以认识的形式
        # pytroch： tensor张量

        if self.train:
            h5f = h5py.File(self.traindbf, 'r')
            key = self.keys[index]
            data = np.array(h5f[key])
            h5f.close()
            return torch.Tensor(data)
        else:
            h5f = h5py.File(self.valdbf, 'r')
            h5f_dirty = h5py.File(self.valdirtydbf, 'r')
            key = self.keys[index]
            data_clean = np.array(h5f[key])
            data_dirty = np.array(h5f_dirty[key])
            h5f.close()
            h5f_dirty.close()
            return torch.Tensor(data_clean), torch.Tensor(data_dirty)