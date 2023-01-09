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
from src.functions import MeasurementError, measure_fn, L2distance_sq, L1distance
from src.utils import Callback
from src.params import params

from torch import optim

import torchvision.utils as tv_utils

from src.experiments import denoise_images_batch, load_test_images, denoise_images_per_iter_adversarial
from src.models.loader import load_classifier
from src.util import attack

# algorithms = {'gd': gd, 'gdm': gdm, 'al': al}

dim_space = {
        'mnist': 28 * 28,
        'celeba': 3 * 64 * 64,
        'lsun': 3 * 64 * 64,
        }

ckpt_mnist = 'pretrained_classifier/mnist/300-300-300-normal.ckpt'

def main(args):
    torch.manual_seed(10)
    algorithms = {
                'gd': alg.gd, 'admm': alg.al, 'eadmm': alg.al,
                'l1eadmm': alg.l1eadmm,
                'linfeadmm': alg.linf_eadmm,
                'pgd': alg.pgd, 'gdm': alg.gdm, 'adam': alg.adam,
                'ladmm': alg.ladmm, 'ladmm2': alg.ladmm2}

    id_ = datetime.datetime.now().strftime('%I%M%p_%b_%d')
    print(id_)

    classifier = load_classifier(model2load=ckpt_mnist, dataset='mnist')
    images, labels = load_test_images('mnist', args.n_images, image_type='real', g=None, return_y=True)
    pgm = attack.PGM(step_size=0.01, threshold=args.std, iter_num=30)
    optimizer = optim.SGD(classifier.parameters(), lr=1)

    def compute_error(ims):
        vals, pred_labs = torch.max(classifier(ims), dim=1)
        correct = torch.sum(pred_labs == labels).float()
        error = (1 - correct / args.n_images) * 100
        return error

    def compute_error_per_checkpoint(ims):
        total_correct = torch.zeros(ims[0].shape[0])
        if ims[0].is_cuda:
            total_correct = total_correct.cuda()
        for i in range(len(ims)):
            vals, pred_labs = torch.max(classifier(ims[i]), dim=1)
            total_correct += (pred_labs == labels[i]).float()

        pdb.set_trace()
        error = (1 - total_correct / args.n_images) * 100
        return error

    print('error on clean images: ', compute_error(images), '%')

    attacked_images = pgm.attack(
            classifier,
            optimizer=optimizer,
            data_batch=images, label_batch=labels)

    print('error on attacked images: ', compute_error(attacked_images), '%')

    denoised_images = denoise_images_per_iter_adversarial(
            true_images=images, images=attacked_images, dataset='mnist',
            norm=args.norm, elu=True, algorithms={k: a for k, a in
            algorithms.items() if k in args.alg}, n_iter=args.n_iter,
            n_points=args.n_points, cuda=args.cuda, seed=args.seed,
            id_=id_, rel_iters=True)

    errors_per_checkpoint = dict()
    for k in args.alg:
        errors_per_checkpoint[k] = compute_error_per_checkpoint(denoised_images[k])

#    outf = 'draft/temp_figures/attacks/' + id_
#
#    if not os.path.exists(outf):
#        os.makedirs(outf)
#
#    src.plots.compare_attacks(images, attacked_images, denoised_final, alg=args.alg, norm=args.norm, outf=outf)


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
    parser.add_argument('--n_images', type=int, default=7,
            help='number of images to denoise.')
    parser.add_argument('--n_points', type=int,
            help='number of points along the trajectory to sample images.')
    parser.add_argument('--std', type=float, default=0.1,
            help='standard deviation for noise')

    parser.add_argument('--alg', type=str, nargs='+', default=['gd'],
            help='name of algorithms to use')

    args = parser.parse_args()
    main(args)
