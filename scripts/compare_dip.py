import pdb
import argparse
import sys, os, datetime, platform
sys.path.insert(0, './')

import copy
import math
import torch
import torchvision.utils as tv_utils
import torchvision.datasets as datasets
import src.plots

from torch import optim
from torchvision import transforms
from src.models import loader
from src.algorithms_dip import gd, gdm, al, adam, rmsprop
from src.functions import MeasurementError, measure_fn
from src.utils import Callback
from src.params_dip import params

# algorithms = {'gd': gd, 'gdm': gdm, 'al': al}
# algorithms = {'gd': gd, 'al': al}
# algorithms = {'gd': gd, 'gdm': gdm, 'adam': adam, 'rmsprop': rmsprop}
# algorithms = {'gd': gd, 'gdm': gdm, 'rmsprop': rmsprop}
algorithms = {'gd': gd, 'gdm': gdm, 'adam': adam, 'al': al}

def main(args):
    torch.manual_seed(args.seed)
    activation = 'elu' if args.elu else 'relu'

    g = loader.load_generator(args.ckpt, args.dataset, input_dim=args.input_dim)

    for param in g.parameters():
        param.requires_grad = True

    mnist_test = datasets.MNIST(
            root='./data', train=False, download=True, transform=transforms.ToTensor())

    x_true = mnist_test[0][0]
    x_true = x_true.view(1, *x_true.shape)

    A = torch.zeros(args.m, x_true.numel()).normal_()# / math.sqrt(args.m)
    b = A @ x_true.view(-1)
    b += torch.zeros_like(b).normal_(std=args.std)

    fn = MeasurementError(A, b)
    z_0 = torch.zeros(args.input_dim).normal_()
    x_0 = g(z_0)

    result = dict()
    fn_values = dict()
    callback = Callback(fn_value='fn_value', infs='infs', beta='beta', sigma='sigma')

    for k, algorithm in algorithms.items():
        g_ = copy.deepcopy(g)
        kw = params[args.dataset + '_' + activation][k]
        result[k] = algorithm(
                fn=fn, g=g_, z_0=z_0, n_iter=args.n_iter, callback=callback, **kw)
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
