import torch
from .base_model import BaseModel
from . import networks
import torch.nn as nn


class AlexNetModel(BaseModel):
    """ This class implements the alex model, for a classify task
    
    The model training requires '--dataset_model ???'
    """

    def __init__(self, opt):
        """ Initialize the AlexNetModel class

        Parameters:
            opt (Option class)  -- stores all the experiment flags and needs to be a subclass of BaseOptions

        """
        BaseModel.__init__(self, opt)

        self.loss_names = ['CrossEntropyLoss']   # specify the training losses you want to print out
        # visual_names = []        # specify the images you want to save or display
        self.model_names = ['alex']
        self.netalex = define_net()
        
        if self.isTrain:
            # define loss functions
            self.loss_func = AlexLoss().to(self.device)
            # initialize optimizers
            self.optimizer = torch.optim.SGD(self.netalex.parameters(), opt.lr, 0.9, 0.0005)
            self.optimizers.append(self.optimizer)

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """ Add new dataset-specific options, and rewrite default values for existing options

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase

        Returns:
            the modified parser

        """
        return parser

    def set_input(self, input):
        """ Unpack input data from the dataloader and perform necessary pre-processing steps

        Parameters:
            input (dict)    -- include the data itself and its metadata information
        """
        self.input = input['data'].to(self.device)
        self.label = input['label'].to(self.device)
        self.image_paths = input['path']

    def forward(self):
        self.output = self.netalex(self.input)

    def backword(self):
        self.loss_CrossEntropyLoss = self.loss_func(self.output, self.label)
        self.loss = self.loss_CrossEntropyLoss
        self.loss.backward()

    def optimize_parameters(self):
        """ Calculate losses, grandients, and update network weights; called in every training iteration """

        self.forward()
        self.set_requires_grad([self.netalex], True)
        self.optimizer.zero_grad()
        self.backword()
        self.optimizer.step()


def define_net(init_type='normal', init_gain=0.01, gpu_ids=[]):
    net = AlexNet()
    return networks.init_net(net, init_type, init_gain, gpu_ids)


class AlexLoss(nn.Module):
    """ Define loss of AlexNet """

    def __init__(self):
        """ Initialize the AlexLoss class

        Parameters:
            target_label (int)  -- label of image

        """
        super(AlexLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, pred_label, target_label):
        """ Calculate loss given AlexNet's output and GT labels

        Parameters:
            pred_label (int)    -- the prediction output from AlexNet
            target_label (int)  -- the GT label of input images

        Returns:
            the calculate loss
        """
        pred_label = pred_label.view(1, -1)
        logsoftmax_func = nn.LogSoftmax(dim=1)
        pred_label = logsoftmax_func(pred_label)
        loss = self.loss(pred_label, target_label)
        return loss


class AlexNet(nn.Module):
    """ Define a AlexNet """

    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 11, 4, padding=2)
        self.conv2 = nn.Sequential(nn.Conv2d(48, 128, 5, 1, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 192, 3, 1, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2))
        self.conv4 = nn.Conv2d(192, 192, 3, 1, padding=1)
        self.conv5 = nn.Sequential(nn.Conv2d(192, 128, 3, 1, padding=1), nn.MaxPool2d(3, 2))
        self.linear1 = nn.Linear(6 * 6 * 128, 2048)
        self.linear2 = nn.Linear(2048, 2048)
        self.linear3 = nn.Linear(2048, 10)

    def forward(self, input):
        """ Standard forward """  # [1, 3, 224, 224]
        conv1_intermediate = self.conv1(input)  # [1, 48, 55, 55]
        conv2_intermediate = self.conv2(conv1_intermediate)  # [1, 128, 27, 27]
        conv3_intermediate = self.conv3(conv2_intermediate)  # [1, 192, 14, 14] why not is 13?
        conv4_intermediate = self.conv4(conv3_intermediate)  # [1, 192, 14, 14]
        conv5_intermediate = self.conv5(conv4_intermediate)  # [1, 128, 6, 6] why is 6?
        view_intermediate = conv5_intermediate.view(-1)
        linear1_intermediate = self.linear1(view_intermediate)
        linear2_intermediate = self.linear2(linear1_intermediate)
        output = self.linear3(linear2_intermediate)
        return output
