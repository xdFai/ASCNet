"""
Different utilities such as orthogonalization of weights, initialization of
loggers, etc

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import subprocess
import math
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from math import exp
from torch.autograd import Variable
from torch.nn import functional as F
from PIL import Image
import random


from torchvision import transforms
import matplotlib.pyplot as plt

def seed_pytorch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weights_init_kaiming(lyr):
    r"""Initializes weights of the model according to the "He" initialization
    method described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    normal distribution.
    This function is to be called by the torch.nn.Module.apply() method,
    which applies weights_init_kaiming() to every layer of the model.
    """
    classname = lyr.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        lyr.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)). \
            clamp_(-0.025, 0.025)
        nn.init.constant(lyr.bias.data, 0.0)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def batch_psnr(img, imclean, data_range):
    r"""
    Computes the PSNR along the batch dimension (not pixel-wise)

    Args:
        img: a `torch.Tensor` containing the restored image
        imclean: a `torch.Tensor` containing the reference image
        data_range: The data range of the input image (distance between
            minimum and maximum possible values). By default, this is estimated
            from the image data-type.
    """
    img_cpu = img.data.cpu().numpy().astype(np.float32)
    imgclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(img_cpu.shape[0]):
        psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], \
                             data_range=data_range)
    return psnr / img_cpu.shape[0]


# def batch_ssim(img, imclean):
#
#     img_cpu = img.data.cpu().numpy().astype(np.float32)
#     imgclean = imclean.data.cpu().numpy().astype(np.float32)
#     ssimall = 0
#     for i in range(img_cpu.shape[0]):
#         ssimall += ssim(img_cpu[i, :, :, :],imgclean[i, :, :, :])
#     return ssimall / img_cpu.shape[0]

def data_augmentation(image, mode):
    r"""Performs dat augmentation of the input image

    Args:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
            0 - no transformation
            1 - flip up and down
            2 - rotate counterwise 90 degree
            3 - rotate 90 degree and flip up and down
            4 - rotate 180 degree
            5 - rotate 180 degree and flip
            6 - rotate 270 degree
            7 - rotate 270 degree and flip
    """
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return np.transpose(out, (2, 0, 1))


def variable_to_cv2_image(varim):
    r"""Converts a torch.autograd.Variable to an OpenCV image

    Args:
        varim: a torch.autograd.Variable
    """
    nchannels = varim.size()[1]
    if nchannels == 1:
        res = (varim.data.cpu().numpy()[0, 0, :] * 255.).clip(0, 255).astype(np.uint8)
    elif nchannels == 3:
        res = varim.data.cpu().numpy()[0]
        res = cv2.cvtColor(res.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        res = (res * 255.).clip(0, 255).astype(np.uint8)
    else:
        raise Exception('Number of color channels not supported')
    return res


def get_git_revision_short_hash():
    r"""Returns the current Git commit.
    """
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()


def init_logger(argdict):
    r"""Initializes a logging.Logger to save all the running parameters to a
    log file

    Args:
        argdict: dictionary of parameters to be logged
    """
    from os.path import join

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    fh = logging.FileHandler(join(argdict.log_dir, 'log.txt'), mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    try:
        logger.info("Commit: {}".format(get_git_revision_short_hash()))
    except Exception as e:
        logger.error("Couldn't get commit number: {}".format(e))
    logger.info("Arguments: ")
    for k in argdict.__dict__:
        logger.info("\t{}: {}".format(k, argdict.__dict__[k]))

    return logger


def init_logger_ipol():
    r"""Initializes a logging.Logger in order to log the results after
    testing a model

    Args:
        result_dir: path to the folder with the denoising results
    """
    logger = logging.getLogger('testlog')
    logger.setLevel(level=logging.INFO)
    fh = logging.FileHandler('out.txt', mode='w')
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def init_logger_test(result_dir):
    r"""Initializes a logging.Logger in order to log the results after testing
    a model

    Args:
        result_dir: path to the folder with the denoising results
    """
    from os.path import join

    logger = logging.getLogger('testlog')
    logger.setLevel(level=logging.INFO)
    fh = logging.FileHandler(join(result_dir, 'log.txt'), mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def normalize(data):
    r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

    Args:
        data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
    """
    return np.float32(data / 255.)


def svd_orthogonalization(lyr):
    r"""Applies regularization to the training by performing the
    orthogonalization technique described in the paper "FFDNet:	Toward a fast
    and flexible solution for CNN based image denoising." Zhang et al. (2017).
    For each Conv layer in the model, the method replaces the matrix whose columns
    are the filters of the layer by new filters which are orthogonal to each other.
    This is achieved by setting the singular values of a SVD decomposition to 1.

    This function is to be called by the torch.nn.Module.apply() method,
    which applies svd_orthogonalization() to every layer of the model.
    """
    classname = lyr.__class__.__name__
    if classname.find('Conv') != -1:
        weights = lyr.weight.data.clone()
        c_out, c_in, f1, f2 = weights.size()
        dtype = lyr.weight.data.type()

        # Reshape filters to columns
        # From (c_out, c_in, f1, f2)  to (f1*f2*c_in, c_out)
        weights = weights.permute(2, 3, 1, 0).contiguous().view(f1 * f2 * c_in, c_out)

        # Convert filter matrix to numpy array
        weights = weights.cpu().numpy()

        # SVD decomposition and orthogonalization
        mat_u, _, mat_vh = np.linalg.svd(weights, full_matrices=False)
        weights = np.dot(mat_u, mat_vh)

        # As full_matrices=False we don't need to set s[:] = 1 and do mat_u*s
        lyr.weight.data = torch.Tensor(weights).view(f1, f2, c_in, c_out). \
            permute(3, 2, 0, 1).type(dtype)
    else:
        pass


def remove_dataparallel_wrapper(state_dict):
    r"""Converts a DataParallel model to a normal one by removing the "module."
    wrapper in the module dictionary

    Args:
        state_dict: a torch.nn.DataParallel state dictionary
    """
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, vl in state_dict.items():
        name = k[7:]  # remove 'module.' of DataParallel
        new_state_dict[name] = vl

    return new_state_dict


def is_rgb(im_path):
    r""" Returns True if the image in im_path is an RGB image
    """
    from skimage.io import imread
    rgb = False
    im = imread(im_path)
    if (len(im.shape) == 3):
        if not (np.allclose(im[..., 0], im[..., 1]) and np.allclose(im[..., 2], im[..., 1])):
            rgb = True
    print("rgb: {}".format(rgb))
    print("im shape: {}".format(im.shape))
    return rgb


def add_noise(img_train, case, noiseIntL):
    noise_S = torch.zeros(img_train.size())
    if case == 3:
        beta1 = np.random.uniform(noiseIntL[0], noiseIntL[1], size=noise_S.size()[0])
        beta2 = np.random.uniform(noiseIntL[0], noiseIntL[1], size=noise_S.size()[0])
        beta3 = np.random.uniform(noiseIntL[0], noiseIntL[1], size=noise_S.size()[0])
        beta4 = np.random.uniform(noiseIntL[0], noiseIntL[1], size=noise_S.size()[0])

        for m in range(noise_S.size()[0]):
            sizeN_S = noise_S[0, 0, :, :].size()
            A1 = np.random.normal(0, beta1[m], sizeN_S[1])  # 一行向量
            A2 = np.random.normal(0, beta2[m], sizeN_S[1])  # 一行向量
            A3 = np.random.normal(0, beta3[m], sizeN_S[1])  # 一行向量
            A4 = np.random.normal(0, beta4[m], sizeN_S[1])  # 一行向量
            # 拉伸
            A1 = np.tile(A1, (sizeN_S[0], 1))
            A2 = np.tile(A2, (sizeN_S[0], 1))
            A3 = np.tile(A3, (sizeN_S[0], 1))
            A4 = np.tile(A4, (sizeN_S[0], 1))
            # add dim
            A1 = np.expand_dims(A1, 0)
            A2 = np.expand_dims(A2, 0)
            A3 = np.expand_dims(A3, 0)
            A4 = np.expand_dims(A4, 0)
            # to tensor
            A1 = torch.from_numpy(A1)
            A2 = torch.from_numpy(A2)
            A3 = torch.from_numpy(A3)
            A4 = torch.from_numpy(A4)
            imgn_train_m = A1 + A2 * img_train[m] + A3 * A3 * img_train[m] + A4 * A4 * A4 * img_train[m] + \
                           img_train[m]
            imgn_train_m_c = torch.clip(imgn_train_m, 0., 1.)
            noise_S[m, :, :, :] = imgn_train_m_c
        imgn_train = noise_S
    return imgn_train




def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / torch.sum(gauss)  # 归一化


# x=gaussian(3,1.5)
# # print(x)
# x=x.unsqueeze(1)
# print(x.shape) #torch.Size([3,1])
# print(x.t().unsqueeze(0).unsqueeze(0).shape) # torch.Size([1,1,1, 3])


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # window_size,1
    # mm:矩阵乘法 t:转置矩阵 ->1,1,window_size,_window_size
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # expand:扩大张量的尺寸，比如3,1->3,4则意味将输入张量的列复制四份，
    # 1,1,window_size,_window_size->channel,1,window_size,_window_size
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# 构造损失函数用于网络训练或者普通计算SSIM值
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def weights_init_kaiming(lyr):
    r"""Initializes weights of the model according to the "He" initialization
    method described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    normal distribution.
    This function is to be called by the torch.nn.Module.apply() method,
    which applies weights_init_kaiming() to every layer of the model.
    """
    classname = lyr.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        lyr.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)). \
            clamp_(-0.025, 0.025)
        nn.init.constant(lyr.bias.data, 0.0)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def normalize(data):
    return data / 255.


def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return False


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:  # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:  # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)  # collect all the images and return
        return return_images


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            # pdb.set_trace()
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class TVloss(nn.Module):
    def __init__(self, TVloss_weight=1):
        super(TVloss, self).__init__()
        self.TVloss_weight = TVloss_weight
        # self.x = x
        # self.y = y

    def forward(self, x, y):
        # x = self.x
        # y = self.y
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        # count_h = self._tensor_size(x[:, :, 1:, :])  # 算出总共求了多少次差
        # count_w = self._tensor_size(x[:, :, :, 1:])
        # h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()

        # x[:,:,1:,:]-x[:,:,:h_x-1,:]就是对原图进行错位，分成两张像素位置差1的图片，第一张图片
        # 从像素点1开始（原图从0开始），到最后一个像素点，第二张图片从像素点0开始，到倒数第二个
        # 像素点，这样就实现了对原图进行错位，分成两张图的操作，做差之后就是原图中每个像素点与相
        # 邻的下一个像素点的差。
        w_tv_x = (x[:, :, :, 1:] - x[:, :, :, :w_x - 1])
        w_tv_y = (y[:, :, :, 1:] - y[:, :, :, :w_x - 1])
        h_tv_x = (x[:, :, 1:, :] - x[:, :, :h_x - 1, :])
        h_tv_y = (y[:, :, 1:, :] - y[:, :, :h_x - 1, :])
        MSE = torch.nn.MSELoss()
        TVloss = (MSE(h_tv_x, h_tv_y) + MSE(w_tv_x, w_tv_y)) * 0.5
        # Drecloss_stripe = torch.pow((w_tv_y - w_tv_x), 2)
        # Drecloss_stripe = (w_tv_y - w_tv_x)**2
        return self.TVloss_weight * TVloss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def WRRGM(A, B):
    DWT = DWTForward(J=3, wave='haar').cuda()
    IDWT = DWTInverse(wave='haar').cuda()
    DMT3_yl, DMT3_yh = DWT(A)
    for tensor in DMT3_yh:
        tensor.zero_()
    out1 = IDWT((DMT3_yl, DMT3_yh))

    DMT3_yl, DMT3_yh = DWT(B)
    for tensor in DMT3_yh:
        tensor.zero_()
    out2 = IDWT((DMT3_yl, DMT3_yh))

    return out1, out2


class MS_SSIM_L1_LOSS(nn.Module):
    """
    Have to use cuda, otherwise the speed is too slow.
    Both the group and shape of input image should be attention on.
    I set 255 and 1 for gray image as default.
    """

    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range=1.0,
                 K=(0.01, 0.03),  # c1,c2
                 alpha=0.025,  # weight of ssim and l1 loss
                 compensation=1.0,  # final factor for total loss
                 cuda_dev=0,  # cuda device choice
                 channel=3):  # RGB image should set to 3 and Gray image should be set to 1
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.channel = channel
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation = compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros(
            (self.channel * len(gaussian_sigmas), 1, filter_size, filter_size))  # 创建了(3*5, 1, 33, 33)个masks
        for idx, sigma in enumerate(gaussian_sigmas):
            if self.channel == 1:
                # only gray layer
                g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            elif self.channel == 3:
                # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
                g_masks[self.channel * idx + 0, 0, :, :] = self._fspecial_gauss_2d(filter_size,
                                                                                   sigma)  # 每层mask对应不同的sigma
                g_masks[self.channel * idx + 1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
                g_masks[self.channel * idx + 2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            else:
                raise ValueError
        self.g_masks = g_masks.cuda(cuda_dev)  # 转换为cuda数据类型

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)
        # Outer product of input and vec2. If input is a vector of size nn and vec2 is a vector of size mm,
        # then out must be a matrix of size (n \times m)(n×m).

    def forward(self, x, y):
        b, c, h, w = x.shape
        assert c == self.channel

        mux = F.conv2d(x, self.g_masks, groups=c, padding=self.pad)  # 图像为96*96，和33*33卷积，出来的是64*64，加上pad=16,出来的是96*96
        muy = F.conv2d(y, self.g_masks, groups=c, padding=self.pad)  # groups 是分组卷积，为了加快卷积的速度

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=c, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=c, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=c, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)
        if self.channel == 3:
            lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]  # 亮度对比因子
            PIcs = cs.prod(dim=1)
        elif self.channel == 1:
            lM = l[:, -1, :, :]
            PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM * PIcs  # [B, H, W]
        # print(loss_ms_ssim)

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, C, H, W]
        # average l1 loss in num channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-self.channel, length=self.channel),
                               groups=c, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation * loss_mix

        return loss_mix.mean()
