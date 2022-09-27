import numpy as np
from os.path import join
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from io import BytesIO
from PIL import Image
import time
from util.util import lprint as print, loop_until_success
import torch


class Visualizer(object):
    def __init__(self, opt, name="main"):
        self.opt = opt
        if opt.isTrain:
            self.name = opt.name
            self.save_dir = join(opt.ckpt, opt.name, 'log', name)
            self.create_writer(logdir=join(self.save_dir))
        else:
            self.name = '%s_%s_%d' % (
                opt.name, opt.dataset, opt.load_iter)
            self.save_dir = join(opt.ckpt, opt.name)
            self.create_writer(logdir=join(self.save_dir, 'ckpts', self.name))

    @loop_until_success
    def create_writer(self, *args, **kwargs):
        self.writer = SummaryWriter(*args, **kwargs)

    @loop_until_success
    def add_image(self, *args, **kwargs):
        self.writer.add_image(*args, **kwargs)

    @loop_until_success
    def add_images(self, *args, **kwargs):
        self.writer.add_images(*args, **kwargs)

    @loop_until_success
    def add_histogram(self, *args, **kwargs):
        self.writer.add_histogram(*args, **kwargs)

    @loop_until_success
    def add_scalar(self, *args, **kwargs):
        self.writer.add_scalar(*args, **kwargs)

    @loop_until_success
    def flush(self):
        self.writer.flush()

    @loop_until_success
    def close(self):
        self.writer.close()

    def display_current_results(self, phase, visuals, iters):
        for k, v in visuals.items():
            v = v.cpu()
            if k == 'depth_map':
                self.process_dmaps(phase, k, v, iters)
            elif k == 'rectifier':
                self.process_rectifiers(phase, k, v, iters)
            else:
                v[0] = torch.clamp(v[0], 0, 1)
                if len(v[0].shape) == 4:
                    v = v[:, 0, 6:9, ...]
                self.add_image('%s/%s' % (phase, k), v[0], iters)
        self.flush()

    def process_dmap(self, dmap):
        buffer = BytesIO()
        plt.figure(1)
        plt.clf()
        plt.axis('off')
        img = plt.imshow(dmap, cmap=plt.cm.hot)
        plt.colorbar()
        plt.savefig(buffer)
        im = np.array(Image.open(buffer).convert('RGB')).transpose(2, 0, 1)
        buffer.close()
        return im / 255

    def process_dmaps(self, phase, k, v, iters):
        dmaps = v[0]
        if len(dmaps) == 1:
            self.add_image('%s/%s'%(phase, k),
                             self.process_dmap(dmaps[0]),
                             iters)
        else:
            self.add_images('%s/%s'%(phase, k),
                              np.stack([self.process_dmap(dmap)\
                                           for dmap in dmaps]),
                              iters)

    def process_rectifiers(self, phase, k, v, iters):
        for i in range(len(v)):
            self.add_histogram('%s/%s'%(phase, i),
                                 v[i],
                                 iters)

    def print_current_losses(self, epoch, iters, losses,
                             t_total, t_data, total_iters):
        message = '(epoch: %d, iters: %d, time: %.3f, forward: %.3f) ' \
                  % (epoch, iters, t_total, t_data)
        for k, v in losses.items():
            message += '%s: %.4e ' % (k, v)
            self.add_scalar('loss/%s' % k, v, total_iters)

        print(message)
