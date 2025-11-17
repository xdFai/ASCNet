import cv2
import numpy as np
import torch
import pywt
import os
import argparse
from numpy import matlib

# args 系列
parser = argparse.ArgumentParser(description='add stripe noise')
parser.add_argument('--cleanfilename', type=str, default=r'D:\18TCSVT\main02\datasets\Test\All_16\image', help="the path of clean image")
parser.add_argument('--generatefilename', type=str, default=r'D:\18TCSVT\main02\datasets\Test\All_16\nosie\Cycle',
                    help="the path of noise image")
opt = parser.parse_args()

if not os.path.isdir(opt.generatefilename):
    os.makedirs(opt.generatefilename)

clist = os.listdir(opt.cleanfilename)
clist.sort()

# 定义噪声大小
noiseB_S = [0.04, 0.08]


case = 0

for i in clist:
    path = os.path.join(opt.cleanfilename, i)
    image = cv2.imread(path)
    img = image[:, :, 0]
    img = np.float32(img / 255.)
    img = torch.Tensor(img)
    noise_S = torch.zeros(img.size())
    sizeN_S = noise_S.size()
    beta = np.random.uniform(noiseB_S[0], noiseB_S[1])
    stripe_max = np.random.normal(0.1, beta)
    stripe_min = -stripe_max


    if case == 0:
        perio = np.random.randint(6, 9)
        stripe = np.zeros([image.shape[0], image.shape[1]])
        stripe[:, 0: perio * int(image.shape[1] / perio)] = matlib.repmat(
            (stripe_max - stripe_min)* np.random.rand(1, perio) + stripe_min, image.shape[0],
            int(image.shape[1] / perio))
        imgn_val = img + stripe



    elif case == 1:
        perio = np.random.randint(6,8)
        stripe = np.zeros([image.shape[0], image.shape[1]])
        stripe[:, 0: perio * int(image.shape[1] / perio)] = matlib.repmat(
            (stripe_max - stripe_min)* np.random.rand(1, perio) + stripe_min, image.shape[0],
            int(image.shape[1] / perio))
        imgn_val = img + stripe

    elif case == 2:
        perio = np.random.randint(9,11)
        stripe = np.zeros([image.shape[0], image.shape[1]])
        stripe[:, 0: perio * int(image.shape[1] / perio)] = matlib.repmat(
            (stripe_max - stripe_min) * np.random.rand(1, perio) + stripe_min, image.shape[0],
            int(image.shape[1] / perio))
        imgn_val = img + stripe

    elif case == 3:
        perio = np.random.randint(6, 9)
        stripe = np.zeros([image.shape[0], image.shape[1]])
        stripe[:, 0: perio * int(image.shape[1] / perio)] = matlib.repmat(
            (stripe_max - stripe_min) * np.random.rand(1, perio) + stripe_min, image.shape[0],
            int(image.shape[1] / perio))
        imgn_val = img + stripe


    noise_img = imgn_val.numpy()

    noise_img_f = noise_img * 255
    noise_img_f = np.clip(noise_img_f, 0, 255)

    cv2.imwrite(os.path.join(opt.generatefilename, i), noise_img_f.astype("uint8"))
