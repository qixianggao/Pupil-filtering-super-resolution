"""This module contains simple helper functions """
import torch
import numpy as np
from PIL import Image
import os
import time
import math
from functools import wraps
import cv2


def calc_psnr_np(sr, hr, scale):
    """ calculate psnr by numpy

    Params:
    sr : numpy.uint8
        super-resolved image
    hr : numpy.uint8
        high-resolution ground truth
    scale : int
        super-resolution scale
    """
    diff = (sr.astype(np.float32) - hr.astype(np.float32)) / 255.
    shave = scale
    if diff.shape[1] > 1:
        convert = np.zeros((1, 3, 1, 1), diff.dtype)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff = diff * (convert) / 256
        diff = diff.sum(axis=1, keepdims=True)

    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = np.power(valid, 2).mean()
    return -10 * math.log10(mse)

def calc_psnr(im1, im2, shave=0):
    """ calculate psnr by torch

    Params:
    im1 : torch.float32
    im2 : torch.float32
    shave : int
    """
    with torch.no_grad():
        im1, im2 = im1.to(torch.float32), im2.to(torch.float32)
        diff = (im1 - im2) / 255.
        if diff.shape[1] > 1:
            diff *= torch.tensor([65.738, 129.057, 25.064],
                    device=diff.device).view(1, 3, 1, 1) / 256
            diff = diff.sum(dim=1, keepdim=True)
        if shave > 0:
            diff = diff[..., shave:-shave, shave:-shave]
        mse = torch.pow(diff, 2).mean()
        return (-10 * torch.log10(mse)).item()


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    lprint(name)
    lprint(mean)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        lprint('shape,', x.shape)
    if val:
        x = x.flatten()
        lprint('mean = %3.3f, min = %3.3f, max = %3.3f, mid = %3.3f, std=%3.3f'
              % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def loop_until_success(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for i in range(10):
            try:
                ret = func(*args, **kwargs)
                break
            except OSError:
                time.sleep(10)
                print('OSError %d/10' % i)
        return ret
    return wrapper

@loop_until_success
def cv2_imread(path, *args, **kwargs):
    img = cv2.imread(path, *args, **kwargs)
    if img is None and os.path.isfile(path):
        raise OSError
    else:
        return img

@loop_until_success
def torch_save(*args, **kwargs):
    torch.save(*args, **kwargs)

@loop_until_success
def torch_load(*args, **kwargs):
    return torch.load(*args, **kwargs)

@loop_until_success
def lprint(*args, **kwargs):
    print(*args, **kwargs)

@loop_until_success
def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

@loop_until_success
def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


color_map = {
    'normal': ['', ''],
    'warning': ['\033[1;37;45m', '\033[0m'],
    'error': ['\033[1;37;41m', '\033[0m'],
}

keeping_times = {
    'normal': 0,
    'warning': 1.5,
    'error': 2,
}


def print_level(s, level='normal'):
    color = color_map.get(level, ['', ''])
    lprint(color[0] + s + color[1])


def prompt(s, width=66, level='normal', keeping_time=None):
    print_level('='*(width+4), level=level)
    ss = s.split('\n')
    if len(ss) == 1 and len(s) <= width:
        print_level('= ' + s.center(width) + ' =', level=level)
    else:
        for s in ss:
            for i in split_str(s, width):
                print_level('= ' + i.ljust(width) + ' =', level=level)
    print_level('='*(width+4), level=level)
    if keeping_time is None:
        keeping_time = keeping_times.get(level, 0)
    if keeping_time > 0:
        time.sleep(keeping_time)


def split_str(s, width):
    ss = []
    while len(s) > width:
        idx = s.rfind(' ', 0, width+1)
        if idx > width >> 1:
            ss.append(s[:idx])
            s = s[idx+1:]
        else:
            ss.append(s[:width])
            s = s[width:]
    if s.strip() != '':
        ss.append(s)
    return ss


def tensor2img(tensor, clip=True, float2int8=False):
    if clip or float2int8:
        tensor = torch.clip(tensor, 0., 1.)
    tensor = tensor.detach().cpu().numpy()
    img = tensor[0].transpose(1, 2, 0)
    if float2int8:
        img = (img * 255).astype(np.uint8)
    return img
