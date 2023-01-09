import pdb
import argparse
import sys, os, datetime, platform, pickle
sys.path.insert(0, './')

import math
import torch
import src.plots
import src.new_plots

from collections import OrderedDict
from itertools import product
from torch import optim
from src.models import loader
from src import algorithms as alg
from src.functions import MeasurementError, measure_fn
from src.utils import Callback
from src.params import params

from src.experiments import recover_images

dim_space = {
        'old_mnist': 28 * 28,
        'mnist': 28 * 28,
        'celeba': 3 * 64 * 64,
        'lsun': 3 * 64 * 64,
        }

algorithms = {
           'gd': alg.gd, 'admm': alg.al, 'eadmm': alg.al,
           'pgd': alg.pgd, 'gdm': alg.gdm, 'adam': alg.adam,
           'ladmm': alg.ladmm, 'ladmm2': alg.ladmm2}


def main(args):
    if args.fun == 'recl2':
        m_vals = [1]
    else:
        m_vals = [int(x * dim_space[args.dataset]) for x in args.m_rel]

    means = OrderedDict()
    mean_error = OrderedDict()
    id_ = datetime.datetime.now().strftime('%I%M%p_%b_%d')
    print(id_)

    image_types = ['synthetic']
    for image_type, m in product(image_types, m_vals):
        #key = image_type + '_' + str(m)
        key = 'm=' + str(m)
        means[key], mean_error[key] = recover_images(
            fun=args.fun, dataset=args.dataset, image_type=image_type,
            m=m, n_iter=args.n_iter, n_images=args.n_images,
            algorithms={k: a for k, a in algorithms.items() if k in args.alg},
            elu=args.elu, id_=id_, cuda=args.cuda, seed=args.seed,
            normalize=args.normalize, restarts=args.restarts, rel_iters=True)

    outf = 'draft/temp_figures/comparison'

    if not os.path.exists(outf):
        os.makedirs(outf)

    perf_plot = src.plots.performance_plot
    error_plot = src.plots.error_plot
    error_vs_m_plot = src.plots.error_vs_m_plot

    try:
        m_vals_plot = [m_vals[i] for i in args.which_m_plot]
    except:
        m_vals_plot = m_vals
    means = src.new_plots.equate_length(means)
    mean_error = src.new_plots.equate_length(mean_error)

    filename = (
        'cs_' + str(args.n_iter) + '_' + args.dataset + '_'
        + '_' + str(args.fun)
        + '_objective_comparison.pdf')

    src.new_plots.iterplot(
        variables=means, xlabel='iterations',
        ylabel='avg. measurement error', xscale='log', yscale='log',
        xlim=None, filename=os.path.join(outf, filename),
        conference='nips', size='one_half')

    filename = (
        'cs_' + str(args.n_iter) + '_' + args.dataset + '_'
        + '_' + str(args.fun)
        + '_error_comparison.pdf')

    #error_plot(dataset=args.dataset, means=mean_error, m_vals=m_vals_plot,
    #        outf=outf, fun=args.fun, n_iter=args.n_iter, smooth=100)
    src.new_plots.iterplot(
        variables=mean_error, xlabel='iterations',
        ylabel='avg. reconst. error', xscale='log', yscale='log',
        xlim=None, filename=os.path.join(outf, filename),
        conference='nips', size='one_half')

    #error_vs_m_plot(dataset=args.dataset, mean_error=mean_error, m_vals=m_vals,
    #    outf=outf, fun=args.fun, dim_space=dim_space[args.dataset],
    #    algs=args.alg, n_iter=args.n_iter)

    data = {'dataset': args.dataset, 'means': means, 'mean_error': mean_error,
            'm_vals': m_vals, 'outf': outf, 'fun': args.fun}

    pickle.dump(data, open(os.path.join(outf, str(args.n_iter) + '_'
        + args.dataset + '_' + args.fun + '_comparison.pickle'), 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1,
            help='random seed')
    parser.add_argument('--z_dist', type=str, default='normal',
            help='distribution of the latent z vector')

    parser.add_argument('--elu', action='store_true',
            help='use ELU activation function instead of ReLU')
    parser.add_argument('--cuda', action='store_true',
            help='use GPU')
    parser.add_argument('--normalize', action='store_true',
            help='use normalized measurements')
    parser.add_argument('--rel_iters', action='store_true',
            help='use normalized number of iterations by complexity')

    parser.add_argument('--dataset', type=str, default='mnist',
            help='name of the dataset to use, one of mnist, cifar10, lsun, celeba.')
    parser.add_argument('--fun', type=str, default='linear',
            help='type of function to use, either "linear" (regression), "phase" (retrieval) or "recl2" (recovery l2)')

    parser.add_argument('--n_iter', type=int, default=4000,
            help='number of iterations for the optimization algorithm.')
    parser.add_argument('--n_images', type=int, default=7,
            help='number of images to recover.')
    parser.add_argument('--m', type=int, default=100,
            help='number of measurements to make')
    parser.add_argument('--std', type=float, default=0.0,
            help='standard deviation for noisy measurements')

    parser.add_argument('--alg', type=str, nargs='+', default=['gd'],
            help='name of algorithms to use')
    parser.add_argument('--m_rel', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.5],
            help='relative values of measurements to use')
    parser.add_argument('--which_m_plot', type=float, nargs='+', default=[2, 3],
            help='relative values of measurements to plot')
    parser.add_argument('--restarts', type=int, default=1,
            help='number of random restarts')

    args = parser.parse_args()
    main(args)
