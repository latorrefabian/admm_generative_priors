import pdb
import argparse
import sys, os, datetime, platform, pickle
sys.path.insert(0, './')

import math
import torch
import src.plots

from itertools import product
from torch import optim
from src.models import loader
from src import algorithms as alg
from src import algorithms_dip as alg_dip
from src.functions import MeasurementError, measure_fn
from src.utils import Callback
from src.params import params

from src.experiments import recover_images_gd

# algorithms = {'gd': gd, 'gdm': gdm, 'al': al}


def main(args):
    if args.prior == 'gen':
        # algorithms = {'gd': alg.gd, 'admm': alg.al, 'gdm': alg.gdm, 'pgd': alg.pgd}
        # algorithms = {'gd': alg.gd, 'admm': alg.al, 'eadmm': alg.al, 'pgd': alg.pgd}
        # algorithms = {'gd': alg.gd, 'admm': alg.al, 'eadmm': alg.al} #'pgd': alg.pgd}
        algorithms = {'gd': alg.gd}
    elif args.prior == 'dip':
        algorithms = {'gd': alg_dip.gd, 'admm': alg_dip.al, 'gdm': alg_dip.gdm, 'pgd': alg_dip.pgd}
    else:
        raise ValueError('uknown prior : ' + args.prior)

    means = dict()
    mean_error = dict()
    id_ = datetime.datetime.now().strftime('%I%M%p_%b_%d')
    print(id_)

    recover_images_gd(
            fun=args.fun, dataset=args.dataset, image_type='synthetic',
            m=args.m, n_iter=args.n_iter, n_images=args.n_images, algorithms=algorithms,
            elu=args.elu, id_=id_, cuda=args.cuda, seed=args.seed, lr=args.lr)

    outf = 'document/figures/comparison'

#    if not os.path.exists(outf):
#        os.makedirs(outf)
#
#    if len(image_types) == 1:
#        perf_plot = src.plots.performance_plot_one_row
#        error_vs_m_plot = src.plots.error_vs_m_plot_one_row
#    elif len(image_types) == 2:
#        perf_plot = src.plots.performance_plot
#        error_vs_m_plot = src.plots.error_vs_m_plot
#
#    perf_plot(dataset=args.dataset, means=means, m_vals=m_vals,
#            outf=outf, fun=args.fun)
#
#    error_vs_m_plot(dataset=args.dataset, mean_error=mean_error, m_vals=m_vals,
#            outf=outf, fun=args.fun, dim_space=dim_space[args.dataset],
#            algs=[x for x in algorithms.keys()])
#
#    data = {'dataset': args.dataset, 'means': means, 'mean_error': mean_error,
#            'm_vals': m_vals, 'outf': outf, 'fun': args.fun}
#    pickle.dump(data, open(os.path.join(outf, args.dataset + '_' + args.fun + '_comparison.pickle'), 'wb'))

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

    parser.add_argument('--dataset', type=str, default='mnist',
            help='name of the dataset to use, one of mnist, cifar10, lsun, celeba.')
    parser.add_argument('--fun', type=str, default='linear',
            help='type of function to use, either "linear" (regression), "phase" (retrieval) or "recl2" (recovery l2)')
    parser.add_argument('--prior', type=str, default='gen',
            help='type of prior either "gen" (generative) or "dip" (deep image prior)')

    parser.add_argument('--n_iter', type=int, default=4000,
            help='number of iterations for the optimization algorithm.')
    parser.add_argument('--n_images', type=int, default=7,
            help='number of images to recover.')
    parser.add_argument('--m', type=int, default=100,
            help='number of measurements to make')
    parser.add_argument('--std', type=float, default=0.0,
            help='standard deviation for noisy measurements')
    parser.add_argument('--lr', type=float, default=0.1,
            help='learning rate')

    args = parser.parse_args()
    main(args)
