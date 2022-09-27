import torch
from torchvision.models import densenet121
from models.base_model import BaseModel
import models.networks as N
import os


class DeBlurDenseModel(BaseModel):

    def __init__(self, opt):
        super(DeBlurDenseModel, self).__init__(opt)
        self.visual_names = ["In", "Out", "GT"]
        self.model_names = ["Deblur", "D"]
        self.loss_names = ["pix", "adv", "dis"]

        self.net_Deblur = FPNDense(opt)
        self.net_D = N.PatchDiscriminator(opt.out_ch, opt.mid_ch)
        if self.isTrain:
            self.optimizer_names = ["Dense_opt", "D_opt"]

        self.In = None
        self.Out = None
        self.GT = None
        self.score_Out = None
        self.score_GT = None
        self.void = torch.zeros((1, 1, 16, 16))

        if self.isTrain:
            self.loss_pix = 0.
            self.loss_adv = 0.
            self.loss_dis = 0.
            self.lambda_adv = opt.lambda_adv

            if opt.optimizer == "Adam":
                self.optimizer_Db = torch.optim.Adam(self.net_Deblur.parameters(), lr=opt.lr)
                self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=opt.lr / 100.)
            self.optimizers = [self.optimizer_Db, self.optimizer_D]

        else:
            self.lambda_adv = 0.

        N.init_net(self.net_Deblur.to(self.device), opt.init_type, gpu_ids=self.gpu_ids)
        N.init_net(self.net_D.to(self.device), opt.init_type, gpu_ids=self.gpu_ids)

    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--fpn_ch', default=256, type=int)
        parser.add_argument('--mid_ch', default=128, type=int)
        parser.add_argument('--out_ch', default=3, type=int)
        parser.add_argument('--lambda_adv', default=0.001, type=float)
        return parser

    def set_input(self, feed):
        self.In = feed["Data"].to(self.device)
        self.GT = feed["GT"].to(self.device)

    def forward(self):
        self.Out = self.net_Deblur(self.In)
        if self.lambda_adv > 0:
            self.score_Out = self.net_D(self.Out)
            self.score_GT = self.net_D(self.GT)
        return self.Out.detach()

    def backward_D(self):
        self.loss_dis = (
            torch.log(self.score_Out + 0.5) - torch.log(self.score_GT + 0.5)
        )
        self.loss_dis.backward(retain_graph=True)

    def backward_Db(self):
        self.loss_pix = N.GradientLoss(reconstruct_metric=torch.nn.L1Loss())(self.Out, self.GT)
        if self.lambda_adv > 0:
            self.loss_adv = - torch.log(self.score_Out + 0.5)
        loss = self.loss_pix + self.loss_adv * self.lambda_adv
        loss.backward()

    def optimize_parameters(self):
        self.iter += 1
        self.forward()
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        if self.lambda_adv > 0:
            if self.iter % 20 == 0:
                self.backward_D()
            self.backward_Db()
        else:
            self.backward_Db()
        for optimizer in self.optimizers:
            #  if load model, set capturable = true
            # optimizer.param_groups[0]['capturable'] = True
            optimizer.step()

    def update_model(self):
        self.epoch += 1
        if self.epoch <= 60:
            self.optimizer_D.param_groups[0]["lr"] = self.opt.lr / 100.
        elif 60 < self.epoch <= 80:
            self.optimizer_Db.param_groups[0]["lr"] = self.opt.lr / 2.
            self.optimizer_D.param_groups[0]["lr"] = self.opt.lr / 200.
        elif 80 < self.epoch <= 100:
            self.optimizer_Db.param_groups[0]["lr"] = self.opt.lr / 10.
            self.optimizer_D.param_groups[0]["lr"] = self.opt.lr / 1000.


class FPNSegHead(torch.nn.Module):

    def __init__(self, in_ch, out_ch, mid_ch):
        super(FPNSegHead, self).__init__()
        self.conv_layer_0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            torch.nn.ReLU(False)
        )
        self.conv_layer_1 = torch.nn.Sequential(
            torch.nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            torch.nn.ReLU(False)
        )

    def forward(self, x):
        x = self.conv_layer_0(x)
        x = self.conv_layer_1(x)
        return x


class FPNDense(torch.nn.Module):

    def __init__(self, opt, pretrained=True):
        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.

        out_ch = opt.out_ch
        mid_ch = opt.mid_ch
        fpn_ch = opt.fpn_ch

        self.fpn = FPN(fpn_ch, pretrained)

        # The segmentation heads on top of the FPN

        self.head_1 = FPNSegHead(fpn_ch, mid_ch, mid_ch)
        self.head_2 = FPNSegHead(fpn_ch, mid_ch, mid_ch)
        self.head_3 = FPNSegHead(fpn_ch, mid_ch, mid_ch)
        self.head_4 = FPNSegHead(fpn_ch, mid_ch, mid_ch)

        self.smooth = torch.nn.Sequential(
            torch.nn.Conv2d(4 * mid_ch, mid_ch, 3, 1, 1),
            torch.nn.BatchNorm2d(mid_ch),
            torch.nn.ReLU(False),
        )
        self.smooth_2 = torch.nn.Sequential(
            torch.nn.Conv2d(mid_ch, mid_ch // 2, 3, 1, 1),
            torch.nn.BatchNorm2d(mid_ch // 2),
            torch.nn.ReLU(False),
        )

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(mid_ch // 2, out_ch, 3, 1, 1),
            torch.nn.Tanh()
        )
        self.upsample_2 = torch.nn.Upsample(scale_factor=2, mode="nearest")
        self.upsample_4 = torch.nn.Upsample(scale_factor=4, mode="nearest")
        self.upsample_8 = torch.nn.Upsample(scale_factor=8, mode="nearest")

    def forward(self, x):
        map0, map1, map2, map3, map4 = self.fpn(x)

        map4 = self.upsample_8(self.head_4(map4))
        map3 = self.upsample_4(self.head_3(map3))
        map2 = self.upsample_2(self.head_2(map2))
        map1 = self.head_1(map1)

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = self.upsample_2(smoothed)
        smoothed = self.smooth_2(smoothed + map0)
        smoothed = self.upsample_2(smoothed)

        final = self.final(smoothed)

        return x + final

    def unfreeze(self):
        for param in self.fpn.parameters():
            param.requires_grad = True


class FPN(torch.nn.Module):

    def __init__(self, mid_ch=256, pretrained=True):
        """
        Creates an `FPN` instance for feature extraction.
        """

        super(FPN, self).__init__()

        self.features = densenet121(pretrained=pretrained).features

        self.encoder_0 = torch.nn.Sequential(
            self.features.conv0,
            self.features.norm0,
            self.features.relu0
        )
        self.pool_0 = self.features.pool0
        self.encoder_1 = self.features.denseblock1
        self.encoder_2 = self.features.denseblock2
        self.encoder_3 = self.features.denseblock3
        self.encoder_4 = self.features.denseblock4
        self.norm = self.features.norm5

        self.trans_1 = self.features.transition1
        self.trans_2 = self.features.transition2
        self.trans_3 = self.features.transition3

        self.lateral_4 = torch.nn.Conv2d(1024, mid_ch, 1, bias=False)
        self.lateral_3 = torch.nn.Conv2d(1024, mid_ch, 1, bias=False)
        self.lateral_2 = torch.nn.Conv2d(512, mid_ch, 1, bias=False)
        self.lateral_1 = torch.nn.Conv2d(256, mid_ch, 1, bias=False)
        self.lateral_0 = torch.nn.Conv2d(64, mid_ch // 2, 1, bias=False)

        self.upsample_2 = torch.nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        enc0 = self.encoder_0(x)
        pooled = self.pool_0(enc0)

        enc1 = self.encoder_1(pooled)
        tr1 = self.trans_1(enc1)

        enc2 = self.encoder_2(tr1)
        tr2 = self.trans_2(enc2)

        enc3 = self.encoder_3(tr2)
        tr3 = self.trans_3(enc3)

        enc4 = self.encoder_4(tr3)
        enc4 = self.norm(enc4)

        lateral4 = self.lateral_4(enc4)
        lateral3 = self.lateral_3(enc3)
        lateral2 = self.lateral_2(enc2)
        lateral1 = self.lateral_1(enc1)
        lateral0 = self.lateral_0(enc0)

        map4 = lateral4
        map3 = lateral3 + self.upsample_2(map4)
        map2 = lateral2 + self.upsample_2(map3)
        map1 = lateral1 + self.upsample_2(map2)

        return lateral0, map1, map2, map3, map4

