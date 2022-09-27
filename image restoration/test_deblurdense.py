from options.test_options import TestOptions
from data.lowlevel_dataset import LowLevelDataset
from models.deblurdense_model import DeBlurDenseModel
import util.util as u
import torch.utils.data as t_data
import torch
import numpy as np
import random
import sys
import time
import os
import cv2 as cv

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
from pytorch_msssim import ms_ssim


sys.argv.extend(["--name", "deblurv2_astigmatism"])
sys.argv.extend(["--data_root", "./dataset"])
sys.argv.extend(["--model", "deblurdense"])
sys.argv.extend(["--dataset", "lowlevel"])
sys.argv.extend(["--data_name", "Aperture45mm"])
sys.argv.extend(["--batch_size", "1"])
sys.argv.extend(["--patch_size", "256"])
sys.argv.extend(["--gpu_ids", "0"])

# sys.argv.extend(["--resume_epoch", "130"])


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test():
    setup_seed(0)
    opt = TestOptions().parse()
    model = DeBlurDenseModel(opt)
    dataset = LowLevelDataset(opt, opt.data_name)
    model.setup(opt)
    dataloader = t_data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    psnr_list = []
    ssim_list = []
    loss_lpips = lpips.LPIPS()
    loss_lpips.cuda(model.device)
    lpips_list = []
    ms_ssim_list = []

    for i, data in enumerate(dataloader):
        model.set_input(data)
        result = model.forward()
        gt = model.GT.detach()
        psnr_list.append(psnr(u.tensor2img(gt, 1, 1), u.tensor2img(result, 1, 1)))
        ssim_list.append(ssim(u.tensor2img(gt, 1, 1), u.tensor2img(result, 1, 1), multichannel=True))
        lpips_list.append(loss_lpips.forward(result, gt).item())
        ms_ssim_list.append(ms_ssim(result, gt, 1).item())

        if i % 10 == 0:
            print(i)
            cv.imwrite(os.path.join(opt.ckpt, opt.name, "results", data["name"][0]),
                       cv.cvtColor(u.tensor2img(result, 1, 1), cv.COLOR_RGB2BGR))

    print(
        "psnr: %.2f, ssim: %.4f, lpips: %.4f, ms-ssim: %.4f" % (
            np.array(psnr_list).mean(),
            np.array(ssim_list).mean(),
            np.array(lpips_list).mean(),
            np.array(ms_ssim_list).mean()
        )
    )
    print("debug")


if __name__ == '__main__':
    test()
