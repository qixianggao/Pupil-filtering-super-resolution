import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


def get_scheduler(optimizer, opt):
    """ Return a learning rate schedule
    
    Parameters:
        optimizer           -- the optimizer of the network
        opt (option class)  -- 
    """
    return []


def init_net(net, init_type='normal', init_gain=0.01, gpu_ids=None):
    """ Initialize a network with registering CPU or GPU device and initializing the network weights
    
    Parameters:
        net (network)       -- the network to be initialized
        init_type (str)     -- the name of an initialization method
        init_gain (float)   -- scaling factor
        gpu_ids (int list)  -- which GPUs the network runs on, e.g. 0,1,2

    Return an initialized network
    """
    if gpu_ids is None:
        gpu_ids = []
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain)
    return net


def init_weights(net, init_type, init_gain):
    """ Initialize network weights
    
    Parameters:
        net (network)       -- the network to be initialized
        init_type (str)     -- the name of an initialization method
        init_gain (float)   -- scaling factor
    """

    def init_func(m):  # define the initialization function
        class_name = m.__class__.__name__
        if hasattr(m, 'weight') and (class_name.find('Conv') != -1 or class_name.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'uniform':
                init.uniform_(m.weight.data, b=init_gain)
            else:
                raise NotImplementedError('[%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif class_name.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class PatchDiscriminator(torch.nn.Module):

    def __init__(self, in_ch, mid_ch):
        super(PatchDiscriminator, self).__init__()
        self.conv_layer_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, mid_ch, 2, 2),
            torch.nn.LeakyReLU(0.2, True)
        )
        self.conv_layer_2 = torch.nn.Sequential(
            torch.nn.Conv2d(mid_ch, mid_ch * 2, 2, 2),
            torch.nn.InstanceNorm2d(mid_ch * 2),
            torch.nn.LeakyReLU(0.2, True)
        )
        self.conv_layer_3 = torch.nn.Sequential(
            torch.nn.Conv2d(mid_ch * 2, mid_ch * 4, 2, 2),
            torch.nn.InstanceNorm2d(mid_ch * 4),
            torch.nn.LeakyReLU(0.2, True)
        )
        self.conv_layer_4 = torch.nn.Sequential(
            torch.nn.Conv2d(mid_ch * 4, 1, 2),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
        x = (x.mean() + torch.rand(1, dtype=x.dtype, device=x.device)) / 2
        return x


class GradientLoss(torch.nn.Module):

    def __init__(self, alpha=0.2, reconstruct_metric=torch.nn.MSELoss()):
        super(GradientLoss, self).__init__()
        self.grad_loss = torch.nn.L1Loss()
        self.reconstruct_loss = reconstruct_metric
        self.alpha = alpha
        self.beta = (1 - self.alpha) / 2

    def __call__(self, pred, target):
        pred_dx = pred[..., 1:, :] - pred[..., :-1, :]
        pred_dy = pred[..., 1:] - pred[..., :-1]
        target_dx = target[..., 1:, :] - target[..., :-1, :]
        target_dy = target[..., 1:] - target[..., :-1]
        loss = (
            self.alpha * self.reconstruct_loss(pred, target) +
            self.beta * self.grad_loss(pred_dx, target_dx) +
            self.beta * self.grad_loss(pred_dy, target_dy)
        )
        return loss


class TotalVariationLoss(torch.nn.Module):

    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def __call__(self, pred):
        pred_dx = pred[..., 1:, :] - pred[..., :-1, :]
        pred_dy = pred[..., 1:] - pred[..., :-1]
        loss = torch.abs(pred_dx).mean() + torch.abs(pred_dy).mean()
        return loss
