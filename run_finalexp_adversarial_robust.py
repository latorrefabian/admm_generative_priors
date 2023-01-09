import pdb
import argparse
import sys, os, datetime, platform, pickle
sys.path.insert(0, './')

import math
import torch
import src.plots
import numpy as np

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

dim_space = {
        'mnist': 28 * 28,
        'celeba': 3 * 64 * 64,
        'lsun': 3 * 64 * 64,
        }

#ckpt_mnist = 'pretrained_classifier/mnist/300-300-300-normal.ckpt'
ckpt_mnist = 'pretrained_classifier/mnist/300-300-300-robust.ckpt'

def main(args):
    torch.manual_seed(args.seed)
    algorithms = {
                'gd': alg.gd, 'admm': alg.al, 'eadmm': alg.al,
                'l1eadmm': alg.l1eadmm,
                'linfeadmm': alg.linf_eadmm_other2,
                'pgd': alg.pgd, 'gdm': alg.gdm, 'adam': alg.adam,
                'ladmm': alg.ladmm, 'ladmm2': alg.ladmm2}

    id_ = datetime.datetime.now().strftime('%I%M%p_%b_%d')
    print(id_)

    classifier = load_classifier(model2load=ckpt_mnist, dataset='mnist')
    images, labels = load_test_images('mnist', args.n_images, image_type='real', g=None, return_y=True)
    pgm = attack.PGM(step_size=0.01, threshold=args.std, iter_num=30)
    optimizer = optim.SGD(classifier.parameters(), lr=1)

    images_minusone_one = images * 2 - 1

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

        error = (1 - total_correct / args.n_images) * 100
        return error.cpu().numpy()

    clean_error = compute_error(images)
    print('error on clean images: (old) ', clean_error, '%')

    clean_error = compute_error(images_minusone_one)
    print('error on clean images: (new)', clean_error, '%')


    attacked_images = pgm.attack(
            classifier,
            optimizer=optimizer,
            data_batch=images, label_batch=labels)

    attacked_images_minusone_one = pgm.attack(
            classifier,
            optimizer=optimizer,
            data_batch=images_minusone_one, label_batch=labels)

    attacked_images_fixed = (attacked_images_minusone_one + 1) / 2

    print('error on attacked images: (old) ', compute_error(attacked_images), '%')
    print('error on attacked images: (new)', compute_error(attacked_images_minusone_one), '%')

    algs_l2 = ['gd', 'adam']
    #algs_l2 = ['adam']
    algs_linf = ['linfeadmm']

    denoised_images_l2 = denoise_images_per_iter_adversarial(
            true_images=images, images=attacked_images_fixed, dataset='mnist',
            norm=2, elu=True, algorithms={k: a for k, a in
            algorithms.items() if k in algs_l2}, n_iter=args.n_iter,
            n_points=args.n_points, cuda=args.cuda, seed=args.seed,
            id_=id_, rel_iters=True)

    denoised_images_linf = denoise_images_per_iter_adversarial(
            true_images=images, images=attacked_images_fixed, dataset='mnist',
            norm=-1, elu=True, algorithms={k: a for k, a in
            algorithms.items() if k in algs_linf}, n_iter=args.n_iter,
            n_points=args.n_points, cuda=args.cuda, seed=args.seed,
            id_=id_, rel_iters=True)

    for k in denoised_images_l2.keys():
        denoised_images_l2[k] = [t.cpu() for t in denoised_images_l2[k]]
        denoised_images_l2[k] = [t * 2 - 1 for t in denoised_images_l2[k]]

    for k in denoised_images_linf.keys():
        denoised_images_linf[k] = [t.cpu() for t in denoised_images_linf[k]]
        denoised_images_linf[k] = [t * 2 - 1 for t in denoised_images_linf[k]]

    l2_adam_error = compute_error_per_checkpoint(denoised_images_l2['adam'])
    l2_gd_error = compute_error_per_checkpoint(denoised_images_l2['gd'])
    linf_eadmm_error = compute_error_per_checkpoint(denoised_images_linf['linfeadmm'])
    baseline = np.repeat(clean_error, len(linf_eadmm_error))
    errors_per_checkpoint = {
            'l2_adam': l2_adam_error,
            'l2_gd': l2_gd_error,
            'linf_al': linf_eadmm_error,
            'base': baseline,
            }

    outf = 'draft/temp_figures/attacks/' + id_
    if not os.path.exists(outf):
        os.makedirs(outf)

    src.plots.iter_plot(x_axis=None,
            x_label='time', y_label='error (%)',
            xscale='linear', yscale='linear',
            outf=outf, name='comparison_error_iter_robust.pdf', legend=True,
            **errors_per_checkpoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1,
            help='random seed')
    parser.add_argument('--restarts', type=int, default=1,
            help='number of random restarts to try')
    parser.add_argument('--z_dist', type=str, default='normal',
            help='distribution of the latent z vector')

    parser.add_argument('--cuda', action='store_true',
            help='use GPU')

    parser.add_argument('--n_iter', type=int, default=4000,
            help='number of iterations for the optimization algorithm.')
    parser.add_argument('--n_images', type=int, default=7,
            help='number of images to denoise.')
    parser.add_argument('--n_points', type=int,
            help='number of points along the trajectory to sample images.')
    parser.add_argument('--std', type=float, default=0.1,
            help='standard deviation for noise')

    args = parser.parse_args()
    main(args)
