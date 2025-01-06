import os
import cv2
import numpy as np
import torch.nn as nn
import argparse
from model.ASCNet import ASCNet
import time
from utils import *
import numpy as np
import torch
import pywt
import torch.nn as nn
import lpips

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Demo")
parser.add_argument("--log_path", type=str,
                    default=r"XXXXXX")
parser.add_argument("--filename", type=str, default=r"XXXX")
parser.add_argument("--save", type=bool, default=False)
opt = parser.parse_args()


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


ssim = SSIM()
loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg')

cleanfilename = os.path.join(opt.filename, 'image')
clclist = ['Gauss', 'Uniform', 'Cycle']



model = ASCNet(1, 1, feats=32)
# model = nn.DataParallel(model).cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load(opt.log_path, map_location='cpu'))


psnr_sum = 0
ssim_sum = 0
lpips_sum = 0


for clc in clclist:
    savepth = os.path.join(opt.filename, 'ASCNet', clc)
    mk = savepth + '\\'
    noisepth = os.path.join(opt.filename, 'noise', clc)
    namelist = os.listdir(cleanfilename)
    namelist.sort()
    model.eval()
    with torch.no_grad():
        for name in namelist:
            # read noise image and process it
            image = cv2.imread(os.path.join(noisepth, name))
            img_np = np.expand_dims(image[:, :, 0], 0)
            img_np = np.float32(img_np / 255.)
            img_tensor = torch.from_numpy(img_np)
            img_tensor = torch.unsqueeze(img_tensor, 0)
            # out, outstripe = model(img_tensor)
            out = model(img_tensor)
            out_val = torch.clip(out, 0., 1.)

            # read clean image
            image2 = cv2.imread(os.path.join(cleanfilename, name))
            img_np2 = np.expand_dims(image2[:, :, 0], 0)
            img_np2562 = np.float32(img_np2 / 255.)
            img_clean = torch.from_numpy(img_np2562)
            # img_clean = torch.unsqueeze(img_clean, 0).cuda()
            img_clean = torch.unsqueeze(img_clean, 0)

            # calculate PSNR and SSIM
            psnr_val = batch_psnr(out_val, img_clean, 1.)
            ssim_val = ssim(img_clean, out_val)
            lpips_val = loss_fn_alex(out_val, img_clean)

            psnr_sum = psnr_sum + psnr_val
            ssim_sum = ssim_sum + ssim_val
            lpips_sum = lpips_sum + lpips_val
            # print(name)
            if opt.save:
                if os.path.exists(mk):
                    pass
                else:
                    os.makedirs(mk)


                out_np = out.data.cpu().numpy()
                # out_np = out.data.numpy()
                out_val = out_np[0, :, :, :]
                out_val = np.transpose(out_val, (1, 2, 0))
                out_val = out_val * 255
                out_valf = np.clip(out_val, 0, 255)
                savepth = os.path.join(opt.filename, 'ASCNet', clc)
                # final=normalization(out_val)
                # out_valf = final * 255
                cv2.imwrite(os.path.join(savepth, name), out_valf.astype("uint8"))

    psnr_val = psnr_sum / len(namelist)
    ssim_val = ssim_sum / len(namelist)
    lpips_val = lpips_sum / len(namelist)

    print("*" * 10 + clc + "*" * 10)
    print("PSNR_sum: %.4f" % psnr_val)
    print("SSIM_sum: %.4f" % ssim_val)
    print("LPIPS_sum: %.4f" % lpips_val)

    psnr_sum = 0
    ssim_sum = 0
    lpips_sum = 0
