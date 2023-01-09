import os, sys, datetime
sys.path.insert(0, './')
import pdb
import argparse, random, os
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as tv_utils
import src.algorithms as alg
import src.plots

import numpy as np

from src.augmented_lagrangian import augmented_lagrangian
from src.utils import load_generator, random_z, compare_recovery
from src.utils import Logger
from functools import partial


def main(args):
    torch.manual_seed(1)
    generator = load_generator(args)

    if args.dataset == 'celeb':
        n_measurements = [1000]
    elif args.dataset == 'mnist':
        n_measurements = [100]

    z_true = random_z(
            distribution=args.z_distribution,
            shape=[1, args.nz, 1, 1], cuda=args.cuda)

    x_true = generator(z_true)

    z_0 = random_z(
            distribution=args.z_distribution,
            shape=z_true.shape, cuda=args.cuda)

    x_0 = generator(z_0)

    lambda_0 = torch.randn(
            x_true[0].numel(), requires_grad=True)

    if args.cuda:
        lambda_0 = lambda_0.cuda()


    results = compare_recovery(x_true=x_true, g=generator, z_0=z_0,
            n_measurements=n_measurements, augmented_lagrangian=augmented_lagrangian, pgd=pgd)

    src.plots.iter_plot(
            x_axis=n_measurements, x_label='\# measurements',
            y_label=r'$\frac{1}{d}||x - x^\star||_2$', outf=args.outf, name='recovery_error.pdf',
            **results['recovery'])

    src.plots.iter_plot(
            x_label='\# iterations',
            y_label=r'$\frac{1}{m}||Ax - b||^2$', scale='loglog', outf=args.outf, name='pgd_loss.pdf',
            **results['optimization']['loss'])

    src.plots.iter_plot(
            x_label='\# iterations',
            y_label=r'$\frac{1}{2}||Ax - b||^2$', scale='loglog', outf=args.outf, name='aug_lagr_error.pdf',
            **results['optimization']['error'])

    src.plots.iter_plot(
            x_label='\# iterations',
            y_label=r'$||G(z) - w||$', scale='loglog', outf=args.outf, name='infeasibility.pdf',
            **results['optimization']['infeasibility'])

    tv_utils.save_image(
            [x_0[0]] + results['images']['pgd'] + [x_true[0]],
            os.path.join(args.outf, 'pgd_recovered.png'), normalize=True)

    tv_utils.save_image(
            [x_0[0]] + results['images']['augmented_lagrangian'] + [x_true[0]],
            os.path.join(args.outf, 'al_recovered.png'), normalize=True)

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
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate for pgd algorithm')

    parser.add_argument('--device', type=str, default=None, help='specify device to use, either gpu or cpu')
    parser.add_argument('--gpu', type=str, default=None, help='specify each GPU to use, default=None')
    parser.add_argument('--cuda', action='store_true', help='wether to use cuda tensors and generator or not')

    args = parser.parse_args()

    id_ = datetime.datetime.now().strftime('%I%M%p_%B_%d_%Y')
    outf = os.path.join(args.outf, id_)

    if not os.path.exists(outf):
        os.makedirs(outf)

    logger = Logger(log_file=os.path.join(outf, 'log.txt'))
    sys.stdout = logger

    args.outf = outf

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

    sys.stdout = logger.terminal
    main(args)
