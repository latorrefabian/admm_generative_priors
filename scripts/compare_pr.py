import pdb
import argparse
import sys, os, datetime, platform
sys.path.insert(0, './')

import math
import torch
import torchvision.utils as tv_utils
import src.plots

from torch import optim
from src.models import loader
from src.algorithms import gd, gdm, al, adam
from src.functions import PhaseRetrieval
from src.utils import Callback
from src.params_pr import params

# algorithms = {'gd': gd, 'gdm': gdm, 'al': al}
# algorithms = {'gd': gd, 'al': al}
algorithms = {'al': al, 'gd': gd, 'gdm': gdm}
# algorithms = {'gd': gd, 'gdm': gdm, 'adam': adam, 'al': al}

def main(args):
    torch.manual_seed(args.seed)
    activation = 'elu' if args.elu else 'relu'

    if args.ckpt is not None:
        ckpt = os.path.join('pretrained_generators', activation, args.ckpt)
    else:
        ckpt = None

    g = loader.load_generator(ckpt, args.dataset, input_dim=args.input_dim)
    g.eval()

    for param in g.parameters():
        param.requires_grad = False

    if args.z_dist == 'normal':
        z_true = torch.zeros(1, args.input_dim).normal_()
    elif args.z_dist == 'uniform':
        z_true = torch.zeros(1, args.input_dim).uniform_()
    else:
        raise ValueError('unknown z distribution: ' + args.z_dist)

    x_true = g(z_true)

    A = torch.zeros(args.m, x_true.numel()).normal_()# / math.sqrt(args.m)
    b = torch.abs(A @ x_true.view(-1))
    b += torch.zeros_like(b).normal_(std=args.std)

    fn = PhaseRetrieval(A, b)
    z_0 = torch.zeros_like(z_true)
    x_0 = g(z_0)

    result = dict()
    fn_values = dict()
    callback = Callback(fn_value='fn_value', infs='infs', beta='beta', sigma='sigma')

    for k, algorithm in algorithms.items():
        kw = params[args.dataset + '_' + activation][k]
        result[k] = algorithm(
                fn=fn, g=g, z_0=z_0, n_iter=args.n_iter, callback=callback, **kw)
        fn_values[k] = callback['fn_value']
        callback.clear()

    id_ = datetime.datetime.now().strftime('%I%M%p_%b_%d')
    outf = os.path.join('document', 'figures', id_ + '_' + args.dataset)
    if not os.path.exists(outf):
        os.makedirs(outf)

    src.plots.iter_plot(
            x_label='iterations',
            y_label=r'$\frac{1}{2}||Ag(z) - b||^2$', xscale='log', yscale='log', outf=outf, name='measurement_error.pdf',
            **fn_values)

    images, filename = [], []
    for k, im in result.items():
        filename.append(k)
        images.append(im[0])

    tv_utils.save_image(
            [x_0[0]] + images + [x_true[0]],
            os.path.join(outf, '-'.join(filename) + '.png'), normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1,
            help='random seed')
    parser.add_argument('--z_dist', type=str, default='normal',
            help='distribution of the latent z vector')

    parser.add_argument('--input_dim', type=int, default=128,
            help='size of the latent z vector')
    parser.add_argument('--elu', action='store_true',
            help='use ELU activation function instead of ReLU')

    parser.add_argument('--dataset', type=str, default='mnist',
            help='name of the dataset to use, one of mnist, cifar10, lsun, celeba.')
    parser.add_argument('--ckpt', type=str,
            help='name of the file containing a checkpoint of the generator')

    parser.add_argument('--n_iter', type=int, default=500,
            help='number of iterations for the optimization algoruithm.')
    parser.add_argument('--m', type=int, default=100,
            help='number of measurements to make')
    parser.add_argument('--std', type=float, default=0.0,
            help='standard deviation for noisy measurements')

    args = parser.parse_args()
    main(args)
