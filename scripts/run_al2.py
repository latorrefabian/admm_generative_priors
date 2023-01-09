import os
import sys
sys.path.insert(0, './')
import pdb
import argparse, random, os
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as tv_utils

import numpy as np

from src.algorithms import augmented_lagrangian
from src.utils import load_generator, random_z


def main(args):
    torch.manual_seed(1)
    generator = load_generator(args)

    z_true = random_z(
            distribution=args.z_distribution,
            shape=[1, args.nz, 1, 1], device=args.device)

    g_z_true = generator(z_true)

    tv_utils.save_image(
            g_z_true[0].data, os.path.join(args.outf, 'true.png'), normalize=True)

    n_measurements = args.measures
    A = torch.zeros([n_measurements, g_z_true[0].numel()], device=args.device).normal_()           # of shape [n, d]
    b = torch.matmul(g_z_true.view(1, -1), torch.transpose(A, 0, 1))                                # of shape [B, n]

    z_0 = random_z(
            distribution=args.z_distribution,
            shape=z_true.shape, device=args.device)

    lambda_0 = torch.randn(
            g_z_true[0].numel(), requires_grad=True, device=args.device)

    w = augmented_lagrangian(
            b=b, A=A, g=generator, z_0=z_0, beta=args.beta,
            gamma=args.gamma, sigma=args.sigma,
            lambda_0=lambda_0, n_iter=args.n_iter)

    tv_utils.save_image(
            w[0].view(g_z_true[0].shape).data, os.path.join(args.outf, 'recovered.png'), normalize=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_distribution', default='uniform', help='uniform | normal, default="uniform"')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector, default=100')
    parser.add_argument('--nc', type=int, default=3, help='number of channels in the generated image, default=3')
    parser.add_argument('--ngf', type=int, default=64, help='number of filters for the conv layers of the generator, default=64')
    parser.add_argument('--g_path', default='pretrained_generators/celeba_dcgan.pt', help='path to generator checkpoint')
    parser.add_argument('--outf', default='document/figures', help='folder to output images and plots, default="document/figures"')
    parser.add_argument('--dataset', type = str, default = 'celeb', help = 'the targeted dataset, default="celeb"')

    parser.add_argument('--beta', type=float, default=0.005, help='beta, default=0.01')
    parser.add_argument('--gamma', type=float, default=3e-5, help='gamma, default=3e-5')
    parser.add_argument('--sigma', type=float, default=0.005, help='sigma, default=3e-5')
    parser.add_argument('--n_iter', type=int, default=10000, help='number of iterations, default=10000')
    parser.add_argument('--measures', type=int, default=2000, help='number of measures provided, default=1000')

    parser.add_argument('--device', type=str, default=None, help='specify device to use, either gpu or cpu')
    parser.add_argument('--gpu', type=str, default=None, help='specify each GPU to use, default=None')

    args = parser.parse_args()

    if args.gpu != None and args.gpu != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print('Use GPU %s' % args.gpu)
    else:
        print('Use default GPU' if args.gpu == None else 'Use CPUs')

    if not os.path.exists(args.outf):
        os.makedirs(args.outf)

    print('**** Arguments ****')
    for arg, value in args._get_kwargs():
        print(arg + ':', value)
    print('*******************')

    main(args)
