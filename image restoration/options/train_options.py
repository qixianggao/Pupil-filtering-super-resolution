from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class include training options.

    It also include shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # visual and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen.')
        parser.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console.')
        
        # network saving and loading parameters
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs.')
        parser.add_argument('--resume', action='store_true', help='continue training')
        
        # training parameters
        parser.add_argument('--epoch_num', type=int, default=100, help='number of epochs with the initial learning rate.')
        parser.add_argument('--optimizer', type=str, default='Adam')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam.')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam.')

        parser.add_argument('--debug', action='store_true')
        self.isTrain = True
        return parser
