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


def img_to_patches(img, win, stride=1):
    r"""Converts an image to an array of patches.

    Args:
        img: a numpy array containing a CxHxW RGB (C=3) or grayscale (C=1)
            image
        win: size of the output patches
        stride: int. stride
    """
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    total_pat_num = patch.shape[1] * patch.shape[2]
    res = np.zeros([endc, win * win, total_pat_num], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            res[:, k, :] = np.array(patch[:]).reshape(endc, total_pat_num)
            k = k + 1
    return res.reshape([endc, win, win, total_pat_num])

def prepare_data(data_path, \
                 val_data_path, \
                 val_data_dirty_path, \
                 patch_size, \
                 stride, \
                 max_num_patches=None, \
                 aug_times=1, \
                 gray_mode=True):
    r"""Builds the training and validations datasets by scanning the
    corresponding directories for images and extracting	patches from them.

    Args:
        data_path: path containing the training image dataset
        val_data_path: path containing the validation image dataset
        patch_size: size of the patches to extract from the images
        stride: size of stride to extract patches
        stride: size of stride to extract patches
        max_num_patches: maximum number of patches to extract
        aug_times: number of times to augment the available data minus one
        gray_mode: build the databases composed of grayscale patches
    """
    # training database
    print('> Training database')
    # scales = [1, 0.9, 0.8, 0.7]
    scales = [1, 0.8, 0.6, 0.4]
    # scales = [1]
    types = ('*.bmp', '*.png')
    files = []
    for tp in types:
        files.extend(glob.glob(os.path.join(data_path, tp)))
    files.sort()

    if gray_mode:
        traindbf = 'train_gray.h5'
        valdbf = 'val_gray.h5'
        valdirtydbf = 'val_dirty_gray.h5'
    else:
        traindbf = 'train_rgb.h5'
        valdbf = 'val_rgb.h5'
        valdirtydbf = 'val_dirty_rgb.h5'

    if max_num_patches is None:
        max_num_patches = 5000000
        print("\tMaximum number of patches not set")
    else:
        print("\tMaximum number of patches set to {}".format(max_num_patches))
    train_num = 0
    i = 0
    with h5py.File(traindbf, 'w') as h5f:
        while i < len(files) and train_num < max_num_patches:
            imgor = cv2.imread(files[i])
            # h, w, c = img.shape
            for sca in scales:
                img = cv2.resize(imgor, (0, 0), fx=sca, fy=sca, \
                                 interpolation=cv2.INTER_CUBIC)
                if not gray_mode:
                    # CxHxW RGB image
                    img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
                else:
                    # CxHxW grayscale image (C=1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.expand_dims(img, 0)
                img = normalize(img)
                # argument
                patches = img_to_patches(img, win=patch_size, stride=stride)
                # data = patches[:, :, :, 1]
                # data = data * 255
                # im = Image.fromarray(data.reshape(64, 64))  # 这是把numpy转化成了PIL
                # im.show()

                print("\tfile: %s scale %.1f # samples: %d" % \
                      (files[i], sca, patches.shape[3] * aug_times))
                for nx in range(patches.shape[3]):
                    data = data_augmentation(patches[:, :, :, nx].copy(), \
                                             np.random.randint(0, 7))
                    h5f.create_dataset(str(train_num), data=data)
                    train_num += 1
                    for mx in range(aug_times - 1):
                        data_aug = data_augmentation(data, np.random.randint(1, 4))
                        h5f.create_dataset(str(train_num) + "_aug_%d" % (mx + 1), data=data_aug)
                        train_num += 1
            i += 1

    # validation database
    print('\n> Validation database')
    files = []
    for tp in types:
        files.extend(glob.glob(os.path.join(val_data_path, tp)))
    files.sort()
    h5f = h5py.File(valdbf, 'w')
    val_num = 0
    for i, item in enumerate(files):
        print("\tfile: %s" % item)
        img = cv2.imread(item)
        if not gray_mode:
            # C. H. W, RGB image
            img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
        else:
            # C, H, W grayscale image (C=1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, 0)
        img = normalize(img)
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    print('\n> Validation dirty database')
    files_dirty = []
    for tp in types:
        files_dirty.extend(glob.glob(os.path.join(val_data_dirty_path, tp)))
    files_dirty.sort()
    h5f = h5py.File(valdirtydbf, 'w')
    val_num_dirty = 0
    for i, item in enumerate(files_dirty):
        print("\tfile: %s" % item)
        img = cv2.imread(item)
        if not gray_mode:
            # C. H. W, RGB image
            img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
        else:
            # C, H, W grayscale image (C=1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, 0)
        img = normalize(img)
        h5f.create_dataset(str(val_num_dirty), data=img)
        val_num_dirty += 1
    h5f.close()

    print('\n> Total')
    print('\ttraining set, # samples %d' % train_num)
    print('\tvalidation set, # samples %d\n' % val_num)
    print('\tvalidation dirty set, # samples %d\n' % val_num_dirty)
