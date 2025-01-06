import os
# from SSIM import *
import cv2
import numpy as np
import torch
import pywt
import torch.nn as nn
import argparse
from model.ASCNet import ASCNet
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Demo")
parser.add_argument("--log_path", type=str,
                    default=r"XXXX.pth")
parser.add_argument("--filename", type=str, default=r"XXXXX")

parser.add_argument("--savepth", type=str, default=r"XXXXX",
                    help='path of result image file')

parser.add_argument("--mk", type=str, default=r"XXXXX/",
                    help='path of result image file')


opt = parser.parse_args()

model = ASCNet(1, 1, feats=32)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(opt.log_path,map_location='cpu'))

namelist = os.listdir(opt.filename)
namelist.sort()

if os.path.exists(opt.mk):
    pass
else:
    os.makedirs(opt.mk)


# def normalization(data):
#     _range = np.max(data) - np.min(data)
#     return (data - np.min(data)) / _range


model.eval()
for name in namelist:
    image = cv2.imread(os.path.join(opt.filename, name))
    img_np = np.expand_dims(image[:, :, 0], 0)
    img_np = np.float32(img_np / 255.)
    img_tensor = torch.from_numpy(img_np)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    # time_start = time.time()
    out = model(img_tensor)
    # time_end = time.time()
    out_np = out.data.cpu().numpy()
    # time_c = time_end - time_start  # 运行所花时间
    # print('time cost', time_c, 's')
    out_val = out_np[0, :, :, :]

    out_val = np.transpose(out_val, (1, 2, 0))


    # Clamp
    out_val = out_val * 255
    out_valf = np.clip(out_val, 0, 255)

    # Normalization
    # final=normalization(out_val)
    # out_valf = final * 255

    cv2.imwrite(os.path.join(opt.savepth, name), out_valf.astype("uint8"))
