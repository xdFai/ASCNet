'''
修改网络
GPU编号
logs 修改名称并 远程新建文件  372
pth名称  345
上传文件

*************************************

'''

import warnings
import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as utils
from torch.utils.tensorboard import SummaryWriter
# from models import FFDNet
from dataset import Dataset
from modelhyper.RHA_PS_B16 import RHA_PS_B16
from model import common
from utils import *
from warmup_scheduler import GradualWarmupScheduler
from torchvision import transforms
import matplotlib.pyplot as plt
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# warnings.filterwarnings('ignore')


def main(args):
    r"""Performs the main training loop
    """
    # Load dataset
    print('> Loading dataset ...')  # 训练和验证都是读的h5文件
    dataset_train = Dataset(train=True, gray_mode=args.gray, shuffle=True)
    dataset_val = Dataset(train=False, gray_mode=args.gray, shuffle=False)
    # 训练数据走的DataLoder  验证数据没有走
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=args.batch_size, shuffle=True)
    print("\t# of training samples: %d\n" % int(len(dataset_train)))

    # Init loggers
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(args.log_dir)
    # **********************************************************************************************
    # build model
    # **********************************************************************************************
    net = RHA_PS_B16(1, 1, feats=16)
    # Define loss
    criterion = nn.MSELoss().cuda()
    ssim = SSIM().cuda()

    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()

    # Optimizer
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.9999), eps=1e-8)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.9999))
    warmup_epochs = 4
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - warmup_epochs,
                                                            eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()
    # noise case

    # case = 3

    start_epoch = 0
    training_params = {}
    training_params['step'] = 0
    training_params['no_orthog'] = args.no_orthog

    # Training
    for epoch in range(start_epoch, args.epochs):
        print("==============RHA_PS_B16==============", epoch, 'lr={:.6f}'.format(scheduler.get_last_lr()[0]))
        psnr_sum = 0
        psnr_val = 0
        ssim_sum = 0
        ssim_val = 0

        # train
        for i, data in enumerate(loader_train, 0):
            # case = random.randint(0, 3)
            case = 3
            # print(case)
            # Pre-training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            # add noise
            imgn_train = add_noise(img_train, case, args.noiseIntL)
            # imgn_train = add_noise2(img_train, case, args.noiseIntL, args.noiseIntS)
            # Create input Variables
            img_train = Variable(img_train.cuda())
            imgn_train = Variable(imgn_train.cuda())

            # Evaluate model and optimize it
            out_train = model(imgn_train)
            # out_train = torch.clamp(model(imgn_train), 0., 1.)
            # torch.clamp(model(imgn_val), 0., 1.)
            # *************************************************************************************************************************
            # loss
            # *************************************************************************************************************************
            loss1 = criterion(out_train, img_train)
            # loss2 = l1(out_train, img_train)
            # loss3 = dre(out_train, img_train)
            # loss4 = tv(out_train - img_train)
            # loss5 = drestr(out_train, img_train)
            loss = loss1
            loss.backward()
            optimizer.step()

            if training_params['step'] % args.save_every == 0:
                # Apply regularization by orthogonalizing filters
                # Results
                model.eval()
                out_train = torch.clip(out_train, 0., 1.)
                psnr_train = batch_psnr(out_train, img_train, 1.)
                ssim_train = ssim(img_train, out_train)
                if not training_params['no_orthog']:
                    model.apply(svd_orthogonalization)

                # Log the scalar values
                writer.add_scalar('loss', loss.item(), training_params['step'])
                writer.add_scalar('PSNR on training data', psnr_train, \
                                  training_params['step'])
                writer.add_scalar('SSIM on training data', ssim_train, \
                                  training_params['step'])
                print("[epoch %d][%d/%d] loss: %.6f PSNR_train: %.4f" % \
                      (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))
            training_params['step'] += 1
        scheduler.step()
        # The end of each epoch

        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                # Validation
                for dataclean, datadirty in dataset_val:
                    datadirty_val = torch.unsqueeze(datadirty, 0)
                    dataclean_val = torch.unsqueeze(dataclean, 0)
                    datadirty_val, dataclean_val = Variable(datadirty_val.cuda()), Variable(dataclean_val.cuda())
                    out_val = torch.clip(model(datadirty_val), 0., 1.)
                    psnr_val = batch_psnr(out_val, dataclean_val, 1.)
                    psnr_sum = psnr_sum + psnr_val
                    ssim_val = ssim(dataclean_val, out_val)
                    ssim_sum = ssim_sum + ssim_val.item()
                psnr_val = psnr_sum / len(dataset_val)
                ssim_val = ssim_sum / len(dataset_val)
                print("\n[epoch %d] PSNR_val: %.4f SSIM_val: %.6f" % (epoch + 1, psnr_val, ssim_val))
                writer.add_scalar('PSNR on validation data', psnr_val, training_params['step'])
                writer.add_scalar('SSIM on validation data', ssim_val, training_params['step'])
                writer.add_scalar('Learning rate', scheduler.get_lr()[0], training_params['step'])

        if epoch == 0:
            best_psnr = psnr_val
            best_ssim = ssim_val

        print("[epoch %d][%d/%d] psnr_avg: %.4f, ssim_avg: %.4f, best_psnr: %.4f, best_ssim: %.6f" %
              (epoch + 1, i + 1, len(dataset_val), psnr_val, ssim_val, best_psnr, best_ssim))

        if psnr_val >= best_psnr:
            best_psnr = psnr_val
            best_ssim = ssim_val
            print('--- save the model @ ep--{} PSNR--{} SSIM--{}'.format(epoch, best_psnr, best_ssim))
            best_psnr_s = format(best_psnr,'.4f')
            best_ssim_s = format(best_ssim,'.6f')
            s = "best_" + "RHA_PS_B16"+"_" + str(best_psnr_s) + "_" + str(best_ssim_s) + ".pth"
            torch.save(model.state_dict(), os.path.join(args.log_dir, s))

        training_params['start_epoch'] = epoch + 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="RHA_PS_B16")
    # ********************************************************************************************************************************
    parser.add_argument("--log_dir", type=str, default="otherlogs/RHA_PS_B16", help='path of log files')
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--epochs", "--e", type=int, default=101, help="Number of total training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--noiseIntL", nargs=2, type=int, default=[0.05, 0.15], help="Noise training interval")
    # parser.add_argument("--noiseIntS", nargs=2, type=int, default=[0, 0.25], help="Noise training interval")
    parser.add_argument("--seed", type=int, default=42, help="Threshold for test")
    parser.add_argument("--gray", default=True, action='store_true',
                        help='train grayscale image denoising instead of RGB')
    parser.add_argument("--no_orthog", action='store_true', help="Don't perform orthogonalization as regularization")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Number of training steps to log psnr and perform orthogonalization")
    argspar = parser.parse_args()

    print("\n#########################################\n"
          "                 RHA_PS_B16               "
          "\n#########################################\n")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    seed_pytorch(argspar.seed)

    main(argspar)
