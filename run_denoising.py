import pdb
import argparse
import sys, os, datetime, platform, pickle
sys.path.insert(0, './')
import math
import torch

from collections import OrderedDict
from itertools import product
from torch import optim

import src.plots
import src.new_plots

from src.models import loader
from src import algorithms as alg
from src.functions import MeasurementError, measure_fn, L2distance_sq, L1distance
from src.utils import Callback
from src.params import params
from src.experiments import denoise_images

# algorithms = {'gd': gd, 'gdm': gdm, 'al': al}

dim_space = {
        'mnist': 28 * 28,
        'celeba': 3 * 64 * 64,
        'lsun': 3 * 64 * 64,
        }

algorithms = {
            'gd': alg.gd, 'admm': alg.al, 'eadmm': alg.al,
            'l1eadmm': alg.l1eadmm,
            'l1eadmm2': alg.l1eadmm2,
            'linfeadmm': alg.linf_eadmm_other,
            'pgd': alg.pgd, 'gdm': alg.gdm, 'adam': alg.adam,
            'ladmm': alg.ladmm, 'ladmm2': alg.ladmm2
        }


def main(args):
    means = OrderedDict()
    mean_error = OrderedDict()
    id_ = datetime.datetime.now().strftime('%I%M%p_%b_%d')
    print(id_)

    # for image_type, m in product(['real', 'synthetic'], m_vals):
    image_types = ['synthetic']
    for image_type, st in product(image_types, args.std):
        key = 'std=' + str(st)
        means[key], mean_error[key] = denoise_images(
            norm=args.norm, dataset=args.dataset,
            std=st, n_iter=args.n_iter, n_images=args.n_images,
            algorithms={k: a for k, a in algorithms.items() if k in args.alg},
            id_=id_, cuda=args.cuda, seed=args.seed, restarts=args.restarts)

    outf = 'draft/temp_figures/comparison'

    if not os.path.exists(outf):
        os.makedirs(outf)

    for st in args.std:
        for key in means.keys():
            if 'l1eadmm' in means[key].keys():
                means[key]['admm'] = means[key]['l1eadmm']
                means[key].pop('l1eadmm')
            if 'linfeadmm' in means[key].keys():
                means[key]['admm'] = means[key]['linfeadmm']
                means[key].pop('linfeadmm')

    alg = args.alg
    if 'l1eadmm' in args.alg:
        alg = [x if x != 'l1eadmm' else 'admm' for x in args.alg]
    if 'linfeadmm' in args.alg:
        alg = [x if x != 'linfeadmm' else 'admm' for x in args.alg]

    if args.dataset == 'mnist' and args.norm == -1:
        for k in means.keys():
            for al in means[k].keys():
                means[k][al] = means[k][al] * 784
                mean_error[k][al] = means[k][al] * 784

    try:
        std_vals_plot = [args.std[i] for i in args.which_std_plot]
    except:
        std_vals_plot = args.std

    if args.norm == 2:
        ylabel_perf = '$\ell_2$ squared error (pixel)'
        ylabel_err = '$\ell_2$ reconst. error (pixel)'
    elif args.norm == 1:
        ylabel_perf = '$\ell_1$ error (pixel)'
        ylabel_err = '$\ell_1$ reconst. error (pixel)'
    elif args.norm == -1:
        ylabel_perf = '$\ell_\infty$ error'
        ylabel_err = '$\ell_\infty$ reconst. error'

    means = src.new_plots.equate_length(means)
    mean_error = src.new_plots.equate_length(mean_error)

    filename = (
        str(args.n_iter) + '_' + args.dataset + '_'
        + str(args.norm) + '_' + str(args.std)
        + '_denoise_perf_comparison.pdf')
    src.new_plots.iterplot(
        variables=means, xlabel='iterations',
        ylabel=ylabel_perf, xscale='log', yscale='log',
        xlim=None, filename=os.path.join(outf, filename),
        conference='nips', size='one_half')

    filename = (
        str(args.n_iter) + '_' + args.dataset + '_'
        + str(args.norm) + '_' + str(args.std)
        + '_denoise_error_comparison.pdf')
    src.new_plots.iterplot(
        variables=mean_error, xlabel='iterations',
        ylabel=ylabel_err, xscale='linear', yscale='linear',
        xlim=None, filename=os.path.join(outf, filename),
        conference='nips', size='one_half')

    #error_vs_noise_plot(dataset=args.dataset, mean_error=means,
    #        std=args.std, outf=outf, norm=args.norm,
    #        dim_space=dim_space[args.dataset], algs=alg,
    #        n_iter=args.n_iter)

    data = {'dataset': args.dataset, 'means': means,
            'std': args.std, 'outf': outf, 'norm': args.norm}
    pickle.dump(data, open(os.path.join(outf, str(args.n_iter) + '_'
        + args.dataset + '_' + str(args.norm) + '_denoise_comparison.pickle'), 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1,
            help='random seed')
    parser.add_argument('--restarts', type=int, default=1,
            help='number of random restarts to try')
    parser.add_argument('--z_dist', type=str, default='normal',
            help='distribution of the latent z vector')

    parser.add_argument('--elu', action='store_true',
            help='use ELU activation function instead of ReLU')
    parser.add_argument('--cuda', action='store_true',
            help='use GPU')

    parser.add_argument('--dataset', type=str, default='mnist',
            help='name of the dataset to use, one of mnist, cifar10, lsun, celeba.')
    parser.add_argument('--norm', type=int, default=2,
            help='norm to use for denoising')

    parser.add_argument('--n_iter', type=int, default=4000,
            help='number of iterations for the optimization algorithm.')
    parser.add_argument('--n_images', type=int, default=1,
            help='number of images to denoise.')
    parser.add_argument('--std', type=float, nargs='+', default=[0.0],
            help='standard deviation for noise')
    parser.add_argument('--which_std_plot', type=float, nargs='+', default=[1, 3],
            help='values of noise to plot')

    parser.add_argument('--alg', type=str, nargs='+', default=['gd'],
            help='name of algorithms to use')

    args = parser.parse_args()
    main(args)

