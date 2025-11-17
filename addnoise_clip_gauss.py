import cv2
import numpy as np
import torch
import pywt
import os
import argparse

# args 系列
parser = argparse.ArgumentParser(description='add stripe noise')
parser.add_argument('--cleanfilename', type=str, default=r'D:\18TCSVT\main02\datasets\IRSTD\Clean', help="the path of clean image")
parser.add_argument('--generatefilename', type=str, default=r'D:\18TCSVT\main02\datasets\IRSTD\noise\Gauss',
                    help="the path of noise image")
opt = parser.parse_args()

if not os.path.isdir(opt.generatefilename):
    os.makedirs(opt.generatefilename)

clist = os.listdir(opt.cleanfilename)
clist.sort()


# 定义高斯噪声大小

# noiseB_S = [0.04, 0.10]
noiseB_S = [0.02, 0.08]

case = 1

for i in clist:
    path = os.path.join(opt.cleanfilename, i)
    image = cv2.imread(path)
    img = image[:, :, 0]
    # img = np.expand_dims(a, axis=0)
    img = np.float32(img / 255.)
    img = torch.Tensor(img)
    # img_val = torch.unsqueeze(img, 0)
    noise_S = torch.zeros(img.size())
    sizeN_S = noise_S.size()
    if case == 0:
        # add stride noise
        # 随机定一个 分布的最大值
        beta = np.random.uniform(noiseB_S[0], noiseB_S[1])
        noise_col = np.random.normal(0, beta, sizeN_S[1])
        S_noise = np.tile(noise_col, (sizeN_S[0], 1))
        S_noise = torch.from_numpy(S_noise)
        imgn_val = S_noise+img

    elif case == 1:
        beta1 = np.random.uniform(noiseB_S[0], noiseB_S[1])  # 一个数字
        beta2 = np.random.uniform(noiseB_S[0], noiseB_S[1])  # 一个数字

        A1 = np.random.normal(0, beta1, sizeN_S[1])  # 一行向量
        A2 = np.random.normal(0, beta2, sizeN_S[1])  # 一行向量
        # 拉伸
        A1 = np.tile(A1, (sizeN_S[0], 1))
        A2 = np.tile(A2, (sizeN_S[0], 1))
        #
        A1 = torch.from_numpy(A1)
        A2 = torch.from_numpy(A2)
        imgn_val = A1 + A2 * img+img

    elif case == 2:
        beta1 = np.random.uniform(noiseB_S[0], noiseB_S[1])  # 一个数字
        beta2 = np.random.uniform(noiseB_S[0], noiseB_S[1])  # 一个数字
        beta3 = np.random.uniform(noiseB_S[0], noiseB_S[1])  # 一个数字

        A1 = np.random.normal(0, beta1, sizeN_S[1])  # 一行向量
        A2 = np.random.normal(0, beta2, sizeN_S[1])  # 一行向量
        A3 = np.random.normal(0, beta3, sizeN_S[1])  # 一行向量
        # 拉伸
        A1 = np.tile(A1, (sizeN_S[0], 1))
        A2 = np.tile(A2, (sizeN_S[0], 1))
        A3 = np.tile(A3, (sizeN_S[0], 1))
        #
        A1 = torch.from_numpy(A1)
        A2 = torch.from_numpy(A2)
        A3 = torch.from_numpy(A3)
        imgn_val = A1 + A2 * img +A3 * A3 * img+img

    elif case == 3:
        beta1 = np.random.uniform(noiseB_S[0], noiseB_S[1])  # 一个数字
        beta2 = np.random.uniform(noiseB_S[0], noiseB_S[1])  # 一个数字
        beta3 = np.random.uniform(noiseB_S[0], noiseB_S[1])  # 一个数字
        beta4 = np.random.uniform(noiseB_S[0], noiseB_S[1])  # 一个数字

        A1 = np.random.normal(0, beta1, sizeN_S[1])  # 一行向量
        A2 = np.random.normal(0, beta2, sizeN_S[1])  # 一行向量
        A3 = np.random.normal(0, beta3, sizeN_S[1])  # 一行向量
        A4 = np.random.normal(0, beta4, sizeN_S[1])  # 一行向量
        # 拉伸
        A1 = np.tile(A1, (sizeN_S[0], 1))
        A2 = np.tile(A2, (sizeN_S[0], 1))
        A3 = np.tile(A3, (sizeN_S[0], 1))
        A4 = np.tile(A4, (sizeN_S[0], 1))
        #
        A1 = torch.from_numpy(A1)
        A2 = torch.from_numpy(A2)
        A3 = torch.from_numpy(A3)
        A4 = torch.from_numpy(A4)
        imgn_val = A1 + A2 * img +A3 * A3 * img +A4 * A4 * A4 * img+img



    noise_img = imgn_val.numpy()

    noise_img_f = noise_img * 255
    noise_img_f = np.clip(noise_img_f, 0, 255)

    cv2.imwrite(os.path.join(opt.generatefilename, i), noise_img_f.astype("uint8"))
