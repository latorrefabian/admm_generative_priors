import argparse
import os
import datetime
import pickle

from collections import OrderedDict
from napalm import algorithms as alg
from src.experiments import new_denoise_images
from src import new_plots


dim_space = {
        'mnist': 28 * 28,
        'celeba': 3 * 64 * 64,
        'lsun': 3 * 64 * 64,
        }


algorithms = OrderedDict([
        ('gd', alg.gd), ('admm', alg.admm),
        ('al', alg.al), ('adam', alg.adam),
])


def main(args):
    fn_mean = OrderedDict()
    reconst_mean = OrderedDict()
    id_ = datetime.datetime.now().strftime('%I%M%p_%b_%d')
    print(id_)

    for st in args.std:
        key = 'std=' + str(st)
        fn_mean[key], reconst_mean[key] = new_denoise_images(
            dataset=args.dataset, norm=args.norm,
            std=st, n_images=args.n_images,
            algorithms={k: a for k, a in algorithms.items() if k in args.alg},
            n_iter=args.n_iter, cuda=args.cuda, seed=args.seed,
            id_=id_, restarts=args.restarts)

    outf = 'draft/temp_figures/comparison'

    if not os.path.exists(outf):
        os.makedirs(outf)

    if args.norm == -1:
        ylabel_mean = r'$\ell_\infty$ measurement error'
        ylabel_rec = r'$\ell_\infty$ reconst. error'
    elif args.norm == 1:
        ylabel_mean = r'$\ell_1$ measurement error (per pixel)'
        ylabel_rec = r'$\ell_1$ reconst. error (per pixel)'
    elif args.norm == 2:
        ylabel_mean = r'$\ell_2$ squared measurement error (per pixel)'
        ylabel_rec = r'$\ell_2$ squared reconst. error (per pixel)'

    if len(fn_mean) == 1:
        plot_fn = new_plots.new_iterplot
        _, fn_mean = fn_mean.popitem()
        _, reconst_mean = reconst_mean.popitem()
    else:
        plot_fn = new_plots.new_iterplots

    fn_mean = new_plots.smooth(fn_mean, k=10)
    reconst_mean = new_plots.smooth(reconst_mean, k=10)

    file = (args.dataset
            + '_' + str(args.n_iter)
            + '_denoising_meas_'
            + str(args.norm)
            + '.pdf'
            )
    plot_fn(
            variables=fn_mean, xscale='log', yscale='log',
            xlabel='iteration (t)', ylabel=ylabel_mean,
            filename=os.path.join(outf, file),
            conference='icml', size='one_column'
            )

    file = (args.dataset
            + '_' + str(args.n_iter)
            + '_denoising_reconst_'
            + str(args.norm)
            + '.pdf'
            )
    plot_fn(
            variables=reconst_mean, xscale='log', yscale='log',
            xlabel='iteration (t)', ylabel=ylabel_rec,
            filename=os.path.join(outf, file),
            conference='icml', size='one_column'
            )

    print(os.path.join(outf, file))

    data = {'dataset': args.dataset, 'mean': fn_mean,
            'reconst_mean': reconst_mean,
            'std': args.std, 'outf': outf, 'norm': args.norm}

    file = os.path.join(
            outf, str(args.n_iter) + '_'
            + args.dataset + '_' + str(args.norm)
            + '_denoise_comparison.pickle'
            )

    pickle.dump(data, open(file, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--seed', type=int, default=1, help='random seed')
    parser.add_argument(
            '--restarts', type=int, default=1,
            help='number of random restarts to try')
    parser.add_argument(
            '--cuda', action='store_true', help='use GPU')
    parser.add_argument(
            '--dataset', type=str, default='mnist',
            help='name of the dataset to use, '
                 'one of mnist, cifar10, lsun, celeba.')
    parser.add_argument(
            '--norm', type=int, default=2, help='norm to use for denoising')
    parser.add_argument(
            '--n_iter', type=int, default=4000,
            help='number of iterations for the optimization algorithm.')
    parser.add_argument(
            '--n_images', type=int, default=7,
            help='number of images to denoise.')
    parser.add_argument(
            '--std', type=float, nargs='+', default=[0.0],
            help='standard deviation for noise')
    parser.add_argument(
            '--which_std_plot', type=float, nargs='+', default=[1, 3],
            help='values of noise to plot')
    parser.add_argument(
            '--alg', type=str, nargs='+', default=['gd'],
            help='name of algorithms to use')

    args = parser.parse_args()
    main(args)

