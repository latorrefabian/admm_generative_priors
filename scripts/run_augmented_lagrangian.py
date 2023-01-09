import os
import sys
sys.path.insert(0, './')
import pdb
import argparse, random, os
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as tv_utils

import numpy as np

from src.models.celeb import DCGAN_G
from src.models.mnist import MNIST_Conv_G
from src.al import solve_augmented_lagrangian

def main(args):

    if args.dataset.lower() in ['celeb',]:
        generator = DCGAN_G(ngpu = 1, nz = args.nz, ngf = args.ngf, nc = args.nc)
    elif args.dataset.lower() in ['mnist',]:
        generator = MNIST_Conv_G(nz = args.nz, ngf = args.ngf, nc = args.nc)
    else:
        raise ValueError('Unrecognized dataset: %s'%args.dataset)

    map_location = 'cuda' if args.gpu != 'cpu' else 'cpu'
    generator.load_state_dict(torch.load(args.g_path, map_location=map_location))
    for param in generator.parameters():
        param.requires_grad = False
    if map_location == 'cuda':
        generator = generator.cuda()
    generator.eval()

    if args.z_distribution.lower() in ['uniform',]:
        z_true = torch.rand([1, args.nz, 1, 1], device=map_location) * 2 - 1
    elif args.z_distribution.lower() in ['normal',]:
        z_true = torch.randn([1, args.nz, 1, 1], device = map_location)
    else:
        raise ValueError('Unrecognized dataset: %s'%args.dataset)
    g_z_true = generator(z_true)

    tv_utils.save_image(
            g_z_true[0].data, os.path.join(args.outf, 'true.png'), normalize=True)

    n_measurements = args.measures
    A = torch.zeros([n_measurements, g_z_true[0].numel()], device=map_location).normal_()           # of shape [n, d]
    b = torch.matmul(g_z_true.view(1, -1), torch.transpose(A, 0, 1))                                # of shape [B, n]

    if args.z_distribution.lower() in ['uniform',]:
        z_0 = (torch.rand(z_true.shape, device = map_location) * 2. - 1.).detach().requires_grad_()
    elif args.z_distribution.lower() in ['normal',]:
        z_0 = (torch.randn(z_true.shape, device = map_location)).detach().requires_grad_()
    else:
        raise ValueError('Unrecognized dataset: %s'%args.dataset)
    alpha = torch.ones([1], requires_grad=True, device=map_location)
    w_0 = generator(z_0).view(1, -1).detach().requires_grad_()
    lamb = torch.randn(np.prod(g_z_true.shape[1:]), requires_grad=True, device=map_location)

    w = solve_augmented_lagrangian(b=b, A=A, g=[generator,], z_0=[z_0,], beta=args.beta, gamma=args.gamma, sigma=args.sigma,
        w=[w_0,], alpha=alpha, lamb=[lamb,], max_iter=args.max_iter, outf=args.outf)
    tv_utils.save_image(
            w[0].view(g_z_true[0].shape).data, os.path.join(args.outf, 'recovered.png'), normalize=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_distribution', default='uniform', help='uniform | normal, default = "uniform"')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector, default = 100')
    parser.add_argument('--nc', type=int, default=3, help='number of channels in the generated image, default = 3')
    parser.add_argument('--ngf', type=int, default=64, help='number of filters for the conv layers of the generator, default = 64')
    parser.add_argument('--g_path', default='pretrained_generators/celeba_dcgan.pt', help='path to generator checkpoint')
    parser.add_argument('--outf', default='document/figures', help='folder to output images and plots, default = "document/figures"')

    parser.add_argument('--dataset', type = str, default = 'celeb', help = 'the targeted dataset, default = "celeb"')

    parser.add_argument('--beta', type=float, default=0.01, help='beta, default=0.01')
    parser.add_argument('--gamma', type=float, default=3e-5, help='gamma, default=3e-5')
    parser.add_argument('--sigma', type=float, default=3e-5, help='sigma, default=3e-5')
    parser.add_argument('--max_iter', type=int, default=1000, help='maximum iterationm, default=10000')
    parser.add_argument('--measures', type=int, default=1000, help='number of measures provided, default=1000')

    parser.add_argument('--gpu', type=str, default=None, help='specify each GPU to use, default=None')

    args = parser.parse_args()

    if args.gpu != None and args.gpu != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print('Use GPU %s'%args.gpu)
    else:
        print('Use default GPU' if args.gpu == None else 'Use CPUs')

    if not os.path.exists(args.outf):
        os.makedirs(args.outf)

    print('**** Arguments ****')
    for arg, value in args._get_kwargs():
        print(arg + ':', value)
    print('*******************')

    main(args)
