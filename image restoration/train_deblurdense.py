from util.visualizer import Visualizer
from options.train_options import TrainOptions
from data.lowlevel_dataset import LowLevelDataset
from models.deblurdense_model import DeBlurDenseModel
import torch.utils.data as t_data
import torch
import numpy as np
import random
import sys
import time
import os
from models import create_model
from data import create_dataset

sys.argv.extend(["--name", "deblurv2_sphere"])
sys.argv.extend(["--data_root", "./dataset"])
sys.argv.extend(["--model", "deblurdense"])
sys.argv.extend(["--dataset", "lowlevel"])
sys.argv.extend(["--data_name", "sphere"])
sys.argv.extend(["--batch_size", "1"])
sys.argv.extend(["--patch_size", "256"])
sys.argv.extend(["--gpu_ids", "0"])
sys.argv.extend(["--print_freq", "100"])
sys.argv.extend(["--display_freq", "2000"])
sys.argv.extend(["--epoch_num", "400"])
sys.argv.extend(["--save_epoch_freq", "10"])

sys.argv.extend(["--init_type", "xavier"])
sys.argv.extend(["--optimizer", "Adam"])
sys.argv.extend(["--lr", "1e-5"])
sys.argv.extend(["--mid_ch", "128"])

# sys.argv.extend(["--resume"])
# sys.argv.extend(["--resume_epoch", "30"])
sys.argv.extend(["--lambda_adv", "0.001"])


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    setup_seed(0)
    opt = TrainOptions().parse()

    model = DeBlurDenseModel(opt)
    dataset = LowLevelDataset(opt, opt.data_name)
    dataloader = t_data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    """
    JNan Pan Aug 9th
    
    The #52 ~ #54 could be replaced by the follow statement:
    
    model = create_model(opt)
    dataset = create_dataset(opt)
    
    But you should add the corresponding options in the command line such as "--shuffle True". 
    See the .options.base_options.BaseOptions for details.
    """
    model.setup(opt)
    visualizer = Visualizer(opt, "main")

    total_iteration = 0
    for epoch in range(model.epoch + 1, opt.epoch_num + 1):
        epoch_time = time.time()
        iteration = 0
        model.update_model()
        for i, data in enumerate(dataloader):
            iter_time = time.time()
            model.set_input(data)
            model.optimize_parameters()
            iteration += 1
            total_iteration += 1

            if iteration % opt.print_freq == 0:
                losses = model.get_current_losses()
                visualizer.print_current_losses(
                    epoch, iteration, losses, time.time() - epoch_time, time.time() - iter_time, total_iteration
                )

            if iteration % opt.display_freq == 0:
                visualizer.display_current_results('train', model.get_current_visuals(), total_iteration)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % epoch)
            model.save_networks(epoch)


if __name__ == '__main__':
    train()
