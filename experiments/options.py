import argparse
import os
import torch

class Options():
    """Options class
    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Base
        self.parser.add_argument('--dataset', default='ecg', help='ecg dataset')
        self.parser.add_argument('--dataroot', default='', help='path to dataset')
        self.parser.add_argument('--batchsize', type=int, default=256, help='input batch size')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
        self.parser.add_argument('--isize', type=int, default=100, help='input sequence size.')
        self.parser.add_argument('--nc', type=int, default=1, help='input sequence channels')
        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--model', type=str, default='anoformer', help='choose model')
        self.parser.add_argument('--outf', default='./output', help='output folder')
        self.parser.add_argument('--pretrainf', default="./output/anoformer/", help='output folder')

        # Transformer
        self.parser.add_argument('--ntoken', type=int, default=400, help='number of vocabs')
        self.parser.add_argument('--emsize', type=int, default=128, help='embedding dimension')
        self.parser.add_argument('--nhid', type=int, default=512, help='dimension of feed forward layer')
        self.parser.add_argument('--nhead', type=int, default=8, help='number of heads')
        self.parser.add_argument('--nlayer_g', type=int, default=9, help='layers of generator')
        self.parser.add_argument('--nlayer_d', type=int, default=6, help='layers of discriminator')
        self.parser.add_argument('--dropout', type=float, default=0.0, help='layers of discriminator')

        # Mask 
        self.parser.add_argument('--mask_rate', type=float, default=0.6, help='rate of mask')
        self.parser.add_argument('--mask_len', type=int, default=16, help='length of a mask')
        
        # Train
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--w_adv', type=float, default=1, help='parameter')
        self.parser.add_argument('--w_gp', type=float, default=10, help='parameter')
        self.parser.add_argument('--folder', type=int, default=0, help='folder index 0-4')
        self.parser.add_argument('--n_aug', type=int, default=0, help='aug data times')

        # Test
        self.parser.add_argument('--istest',action='store_true',help='train model or test model')
        self.parser.add_argument('--pretrain',action='store_true',help='train model or test model')
        self.parser.add_argument('--threshold', type=float, default=0.05, help='threshold score for anomaly')

        self.opt = None


    def parse(self):
        """ Parse Arguments.
        """
        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        # print('------------ Options -------------')
        # for k, v in sorted(args.items()):
        #     print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ----------------')

        # save to the disk
        self.opt.name = "%s/%s" % (self.opt.model, self.opt.dataset)
        expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
