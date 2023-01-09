import pdb
import argparse
import sys, os, datetime, platform
sys.path.insert(0, './')

from itertools import product
from src import algorithms as alg

from src.experiments import parameter_tuning
from src.models import loader

# algorithms = {'gd': gd, 'gdm': gdm, 'al': al}



def main(args):
    # algorithms = {'gd': alg.gd, 'admm': alg.al, 'gdm': alg.gdm, 'pgd': alg.pgd}
    algorithms = {'gd': alg.gd, 'admm': alg.al, 'eadmm': alg.al, 'pgd': alg.pgd}

    #for image_type, alg_name in product(['real', 'synthetic'], args.alg):
    for image_type, alg_name in product(['synthetic'], args.alg):
        result = parameter_tuning(
            dataset=args.dataset, fun=args.fun, alg_name=alg_name, algorithm=algorithms[alg_name],
            seed=args.seed, image_type=image_type,
            m=args.m, n_iter=args.n_iter, n_images=args.n_images, elu=args.elu, cuda=args.cuda, normalize=args.normalize)

        with open('param_tuning.log', 'a') as logfile:
            logfile.write(args.fun + '-' + args.dataset + '-' + alg_name + ':'
                    + 'best tuning for ' + image_type + ' images: ' + str(result))
            logfile.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1,
            help='random seed')
    parser.add_argument('--m', type=int, default=200,
            help='number of linear measurements')
    parser.add_argument('--z_dist', type=str, default='normal',
            help='distribution of the latent z vector')
    parser.add_argument('--alg', type=str, nargs='+', default=['gd'],
            help='name of algorithm to tune')

    parser.add_argument('--elu', action='store_true',
            help='use ELU activation function instead of ReLU')
    parser.add_argument('--cuda', action='store_true',
            help='use GPU')
    parser.add_argument('--normalize', action='store_true',
            help='normalize measurements')

    parser.add_argument('--dataset', type=str, default='mnist',
            help='name of the dataset to use, one of mnist, cifar10, lsun, celeba.')
    parser.add_argument('--fun', type=str, default='linear',
            help='type of function to use, either "linear" (regression) or "phase" (retrieval).')

    parser.add_argument('--n_iter', type=int, default=4000,
            help='number of iterations for the optimization algorithm.')
    parser.add_argument('--n_images', type=int, default=7,
            help='number of images to recover.')
    parser.add_argument('--std', type=float, default=0.0,
            help='standard deviation for noisy measurements')

    args = parser.parse_args()
    main(args)
