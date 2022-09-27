import torch
import torch.fft as tfft
import torch.nn.functional as F
from models.base_model import BaseModel
import models.networks as N
from scipy.fftpack import idct
import numpy as np
from models.deblurdense_model import FPNDense
import os


class FDeconvModel(BaseModel):

    def __init__(self, opt):
        super(FDeconvModel, self).__init__(opt)
        self.visual_names = ["In", "Out", "GT", "Kernel", "Out_DB"]
        self.model_names = ["FDeconv", "G", "D"]
        self.loss_names = ["pix_db", "pix_dn", "adv", "dis"]

        self.net_FDeconv = FDeconvDeblurNet(opt)
        self.net_G = FPNDense(opt)
        self.net_D = N.PatchDiscriminator(3, opt.mid_ch)
        if self.isTrain:
            self.optimizer_names = ["FDeconv_opt", "G_opt", "D_opt"]

        self.c = -1
        self.In = None
        self.Out = None
        self.Out_DB = None
        self.GT = None
        self.Kernel = None
        self.score_Out = None
        self.score_GT = None
        self.void = torch.zeros((1, 1, 16, 16))

        if self.isTrain:
            self.loss_pix_db = 0.
            self.loss_pix_dn = 0.
            self.loss_adv = 0.
            self.loss_dis = 0.
            self.loss = 0.
            self.lambda_adv = opt.lambda_adv

            if opt.optimizer == "Adam":
                self.optimizer_DB = torch.optim.Adam(self.net_FDeconv.parameters(), lr=opt.lr / 100.)
                self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=opt.lr)
                self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=opt.lr / 100.)
            self.optimizers = [self.optimizer_DB, self.optimizer_G, self.optimizer_D]
        else:
            self.lambda_adv = 0.

        N.init_net(self.net_FDeconv.to(self.device), opt.init_type, gpu_ids=self.gpu_ids)
        N.init_net(self.net_G.to(self.device), opt.init_type, gpu_ids=self.gpu_ids)
        N.init_net(self.net_D.to(self.device), opt.init_type, gpu_ids=self.gpu_ids)
        self.net_FDeconv.f_deconv_block.load_state_dict(torch.load(os.path.join(opt.ckpt, opt.name, "pretrained.pth")))

    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--stage_num', default=7, type=int)
        parser.add_argument('--mid_ch', default=128, type=int)
        parser.add_argument('--in_ch', default=1, type=int)
        parser.add_argument('--out_ch', default=1, type=int)
        parser.add_argument('--fpn_ch', default=256, type=int)
        parser.add_argument('--lambda_adv', default=0.001, type=float)
        return parser

    def set_input(self, feed):
        if self.c == -1:
            self.c = np.random.randint(0, 3)
        self.In = feed["Data"][:, None, self.c, ...].to(self.device)
        self.GT = feed["GT"][:, None, self.c, ...].to(self.device)
        self.Kernel = feed["Kernel"].to(self.device)
        self.c = -1

    def forward(self):
        self.Out_DB = self.net_FDeconv(self.In, self.Kernel)
        self.Out = self.net_G(
            torch.cat([self.Out_DB.detach(), self.In, self.In], 1)
        ).mean(1, keepdim=True)
        if self.lambda_adv > 0:
            self.score_Out = self.net_D(self.Out.repeat(1, 3, 1, 1))
            self.score_GT = self.net_D(self.GT.repeat(1, 3, 1, 1))
        self.Kernel = self.Kernel / self.Kernel.max()
        return self.Out

    def backward_DB(self):
        self.loss_pix_db = torch.nn.L1Loss()(self.Out_DB, self.GT)
        self.loss = self.loss_pix_db
        self.loss.backward()

    def backward_DN(self):
        self.loss_pix_dn = N.GradientLoss(reconstruct_metric=torch.nn.L1Loss())(self.Out, self.GT)
        if self.lambda_adv > 0:
            self.loss_adv = - torch.log(self.score_Out + 0.5)
        self.loss = self.loss_pix_dn + self.loss_adv * self.lambda_adv
        self.loss.backward()

    def backward_D(self):
        self.loss_dis = (
            torch.log(self.score_Out + 0.5) - torch.log(self.score_GT + 0.5)
        )
        self.loss_dis.backward(retain_graph=True)

    def optimize_parameters(self):
        self.iter += 1
        self.forward()
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        if self.lambda_adv > 0:
            if self.iter % 20 == 0:
                self.backward_D()
        self.backward_DN()
        self.backward_DB()
        for optimizer in self.optimizers:
            optimizer.step()

    def update_model(self):
        self.epoch += 1
        if self.epoch > 10:
            for para in self.net_FDeconv.f_deconv_block.parameters():
                para.requires_grad = False
        if 20 < self.epoch <= 30:
            self.optimizer_D.param_groups[0]["lr"] = self.opt.lr / 10.
        elif 30 < self.epoch <= 40:
            self.optimizer_DB.param_groups[0]["lr"] = self.opt.lr / 2.
            self.optimizer_G.param_groups[0]["lr"] = self.opt.lr / 2.
            self.optimizer_D.param_groups[0]["lr"] = self.opt.lr / 20.
        elif 40 < self.epoch <= 50:
            self.optimizer_DB.param_groups[0]["lr"] = self.opt.lr / 10.
            self.optimizer_G.param_groups[0]["lr"] = self.opt.lr / 10.
            self.optimizer_D.param_groups[0]["lr"] = self.opt.lr / 100.


class FDeconvDeblurNet(torch.nn.Module):

    def __init__(self, opt):
        super(FDeconvDeblurNet, self).__init__()
        self.dn_block = DnCNNModule(opt)
        self.f_deconv_block = FDeconvNet(opt)

    def forward(self, y, k):
        y = self.dn_block(y) + y
        x = self.f_deconv_block(y, k)
        return x


class FDeconvNet(torch.nn.Module):

    def __init__(self, opt):
        super(FDeconvNet, self).__init__()
        self.stage_num = opt.stage_num
        self.we_block = torch.nn.Sequential(
            torch.nn.Conv2d(opt.in_ch, opt.mid_ch, 3, 1, 1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(opt.mid_ch, opt.mid_ch, 3, 1, 1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(opt.mid_ch, opt.mid_ch, 3, 1, 1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(opt.mid_ch, 1, 1),
        )
        self.f_deconv_block = torch.nn.Sequential(*[
            FourierDeconvolution(opt.in_ch, opt.out_ch, opt.mid_ch, i) for i in range(opt.stage_num)
        ])

    def forward(self, y, k):
        _, _, W, H = k.shape
        y = F.pad(y, (W // 8, W // 8, H // 8, H // 8), "replicate")
        k = F.pad(k, (W // 8, W // 8, H // 8, H // 8), "replicate")
        w = self.we_block(y)
        x = self.f_deconv_block[0](y, y, k, w)
        for i in range(1, self.stage_num):
            x = self.f_deconv_block[i](x, y, k, w)
        x_db = x[..., W // 8:-W // 8, H // 8:-H // 8]
        return x_db


class FourierDeconvolution(torch.nn.Module):

    def __init__(self, in_ch, out_ch, mid_ch, stage, filter_size=5):
        super(FourierDeconvolution, self).__init__()
        self.stage = stage
        self.filter_size = filter_size
        self.f = torch.matmul(torch.Tensor(self.dct_filters()), torch.eye(self.filter_size ** 2 - 1))[None, None, ...]
        self.f = torch.nn.Parameter(self.f, requires_grad=True)
        self.conv_block = ConvBlock(in_ch, out_ch, mid_ch)

    def dct_filters(self):
        filter_size = self.filter_size ** 2
        filters = np.zeros((filter_size, filter_size - 1), np.float32)
        for i in range(1, filter_size):
            d = np.zeros(filter_size, np.float32)
            d.flat[i] = 1
            filters[:, i - 1] = idct(idct(d, norm='ortho').T, norm='ortho').real.flatten()
        return filters

    @staticmethod
    def psf2otf(psf, img_size):
        top_left = psf[..., :psf.shape[-2] // 2, :psf.shape[-1] // 2]
        top_right = psf[..., psf.shape[-2] // 2:, :psf.shape[-1] // 2]
        bottom_left = psf[..., :psf.shape[-2] // 2, psf.shape[-1] // 2:]
        bottom_right = psf[..., psf.shape[-2] // 2:, psf.shape[-1] // 2:]

        zeros_top = torch.zeros(
            (psf.shape[0], psf.shape[1], img_size[-2] - psf.shape[-2], psf.shape[-1] // 2),
            dtype=psf.dtype, device=psf.device
        )
        zeros_bottom = torch.zeros_like(zeros_top)
        zeros_mid = torch.zeros(
            (psf.shape[0], psf.shape[1], img_size[-2], img_size[-1] - psf.shape[-1]),
            dtype=psf.dtype, device=psf.device
        )

        top = torch.cat([top_left, zeros_top, top_right], -2)
        bottom = torch.cat([bottom_left, zeros_bottom, bottom_right], -2)
        psf_shift = torch.cat([top, zeros_mid, bottom], -1)

        otf = tfft.fft2(psf_shift)
        return otf

    def forward(self, x, y, k, w):

        w = torch.mean(w)
        x_feature = tfft.fft2(self.conv_block(x))

        f = self.psf2otf(self.f, x.shape)
        f = (torch.abs(f) ** 2).sum(1)[:, None, ...]

        F_k = self.psf2otf(k, x.shape)

        if self.stage > 0:
            mask = torch.zeros_like(x)
            mask[..., k.shape[-2]:k.shape[-1], k.shape[-2]:k.shape[-1]] = 1.
            x = tfft.ifft2(tfft.fft2(x) * F_k).real
            x = tfft.fftshift(x, (-2, -1))
            x = (1 - mask) * y + mask * x
            x = tfft.fft2(x) * torch.conj(F_k)
        else:
            x = tfft.fft2(y) * torch.conj(F_k)

        x = tfft.ifft2((w * x + x_feature) / (w * torch.abs(F_k) ** 2 + f)).real
        x = tfft.fftshift(x, (-2, -1))
        return x


class ConvBlock(torch.nn.Module):

    def __init__(self, in_ch, out_ch, mid_ch):
        super(ConvBlock, self).__init__()
        self.conv_in_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            torch.nn.ReLU(True)
        )
        self.conv_block = torch.nn.Sequential(*[
            torch.nn.Sequential(
                torch.nn.Conv2d(mid_ch, mid_ch, 3, 1, 1),
                torch.nn.ReLU(True)
            )] * 4
        )
        self.conv_out_layer = torch.nn.Conv2d(mid_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv_in_layer(x)
        x = self.conv_block(x)
        x = self.conv_out_layer(x)
        return x


class DnCNNModule(torch.nn.Module):

    def __init__(self, opt):
        super(DnCNNModule, self).__init__()

        self.conv_in_layer = torch.nn.Sequential(
            torch.nn.Conv2d(opt.in_ch, opt.mid_ch // 4, 7, 1, 3),
            torch.nn.BatchNorm2d(opt.mid_ch // 4),
            torch.nn.Conv2d(opt.mid_ch // 4, opt.mid_ch, 3, 1, 1),
            torch.nn.ReLU(True)
        )
        self.conv_list = torch.nn.Sequential(*[
            torch.nn.Conv2d(opt.mid_ch, opt.mid_ch, 3, 1, 1),
            torch.nn.BatchNorm2d(opt.mid_ch),
            torch.nn.Conv2d(opt.mid_ch, opt.mid_ch, 3, 1, 1),
            torch.nn.ReLU(True)
        ] * 7)
        self.conv_out_layer = torch.nn.Conv2d(opt.mid_ch, opt.out_ch, 1)

    def forward(self, x):
        x = self.conv_in_layer(x)
        x = self.conv_list(x)
        x = self.conv_out_layer(x)
        return x


if __name__ == '__main__':
    class Option(object):

        def __init__(self):
            self.isTrain = False
            self.data_root = "../Datasets"
            self.data_name = "Luna16"
            self.patch_size = 256
            self.batch_size = 1
            self.gpu_ids = [1]
            self.ckpt = "../ckpt"
            self.name = "f_deconv"
            self.in_ch = 1
            self.mid_ch = 128
            self.out_ch = 1
            self.init_type = "xavier"
            self.optimizer = "Adam"
            self.stage_num = 7
            self.lr = 1e-4
            self.isTrain = True
            self.lambda_adv = 0.001
            self.device = "cuda:1"
            self.fpn_ch = 256

    import matplotlib.pyplot as plt
    import cv2 as cv
    import os
    import util.util as u

    y = cv.cvtColor(cv.imread(os.path.join("..", "dataset", "voc", "Train", "Blur", "2007_001717.jpg")), cv.COLOR_BGR2RGB)
    x = cv.cvtColor(cv.imread(os.path.join("..", "dataset", "voc", "Train", "GT", "2007_001717.jpg")), cv.COLOR_BGR2RGB)
    k = np.load(os.path.join("..", "dataset", "voc", "Train", "Kernel", "2007_001717.npy"))
    k = torch.from_numpy(k)

    y_ = torch.from_numpy((y / 255.).astype(np.float32).transpose(2, 0, 1)[None, ...])
    x_ = torch.from_numpy((x / 255.).astype(np.float32).transpose(2, 0, 1)[None, ...])
    net = FDeconvModel(Option())
    net.net_FDeconv.load_state_dict(torch.load(os.path.join(".", "f_deconv.pth")))
    for i in range(5000):

        data = {"Data": y_, "GT": x_, "Kernel": k}
        # data = {"Data": y_[:, 0, None, ...], "GT": x_[:, 0, None, ...], "Kernel": k}

        net.update_model()
        net.set_input(data)
        net.optimize_parameters()
        print("epoch: %d" % i, net.loss_pix_db, net.loss_pix_dn, net.Out.mean())

        if i % 500 == 0:
            z = torch.ones_like(x_)
            z_db = torch.ones_like(x_)
            for c in range(3):
                net.set_input({"Data": y_, "GT": x_, "Kernel": k})
                z[:, c, ...] = torch.clip(net.forward(), 0., 1.)
                z_db[:, c, ...] = net.Out_DB

            plt.subplot(2, 3, 1)
            plt.imshow(u.tensor2img(k))
            plt.subplot(2, 3, 2)
            plt.imshow(u.tensor2img(y_))
            plt.subplot(2, 3, 3)
            plt.imshow(u.tensor2img(x_))
            plt.subplot(2, 3, 4)
            plt.imshow(u.tensor2img(z))
            plt.subplot(2, 3, 5)
            plt.imshow(u.tensor2img(z_db))
            plt.show()

            from skimage.metrics import peak_signal_noise_ratio as psnr
            print(psnr(u.tensor2img(x_), u.tensor2img(z)))
            print("debug")

    # torch.save(net.net_FDeconv.state_dict(), os.path.join(".", "f_deconv.pth"))
