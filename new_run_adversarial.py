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
from src.models.loader import load_classifier, load_generator
from src.utils import Callback
from src import params
from napalm import algorithms as alg

from torch import optim

import torchvision.utils as tv_utils

from src.experiments import load_test_images, denoise_adversarial
from src.util import attack

dim_space = {
        'mnist': 28 * 28,
        'fmnist': 28 * 28,
        'celeba': 3 * 64 * 64,
        'lsun': 3 * 64 * 64,
        }

algorithms = {
        'gd': alg.gd, 'admm': alg.admm,
        'al': alg.al, 'adam': alg.adam
}

ckpts = {
        'mnist': 'pretrained_classifier/mnist_0_1/300-300-300-0.0.ckpt',
        'fmnist': 'pretrained_classifier/fmnist_0_1/fmnist_convnet-fc-10class-ELU_e40_adam.pt',
        'celeba2': 'pretrained_classifier/celeba_-1_1/celeba_convnet-fc-2class-ELU_e40_adam.pt'
}

ckpts_gen = {
        'mnist': 'pretrained_classifier/mnist_0_1/300-300-300-0.0.ckpt',
        'fmnist': 'pretrained_generators/elu/fmnist_dc-gen-ELU-Sigmoid_e81_adam_normal.pt',
        'celeba2': 'pretrained_generators/elu/celeba_dc-gen-batchnorm-ELU-Tanh_e50_adam_normal.pt'
}

def main(args):
    torch.manual_seed(args.seed)

    id_ = datetime.datetime.now().strftime('%I%M%p_%b_%d')
    print(id_)

    classifier = load_classifier(model2load=ckpts[args.dataset], dataset=args.dataset)
    images, labels = load_test_images(args.dataset, args.n_images, image_type='real', g=None, return_y=True)
    pgm = attack.PGM(step_size=0.01, threshold=args.std, iter_num=30)
    optimizer = optim.SGD(classifier.parameters(), lr=1)

    #g = load_generator(
    #    ckpts_gen[args.dataset], args.dataset,
    #    input_dim=params.input_dims[args.dataset], elu=True)

    #g.eval()
    #test_z0 = torch.zeros(64, params.input_dims[args.dataset]).normal_()
    #test_images = g(test_z0)

    #tv_utils.save_image(
    #        test_images,
    #        args.dataset + '_gen_images.png', normalize=True)

    #pdb.set_trace()
    def compute_error(ims):
        vals, pred_labs = torch.max(classifier(ims), dim=1)
        correct = torch.sum(pred_labs == labels).float()
        error = (1 - correct / args.n_images) * 100
        return error.cpu().numpy()

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
    print('error on clean images: ', clean_error, '%')

    noisy_images = pgm.attack(
            classifier,
            optimizer=optimizer,
            data_batch=images, label_batch=labels)

    noisy_images = torch.clamp(noisy_images, 0, 1).detach()

    print('error on attacked images: ', compute_error(noisy_images), '%')

    denoised_images_l2 = denoise_adversarial(
            images=images, noisy_images=noisy_images, std=args.std, dataset=args.dataset,
            norm=2, algorithms={k: a for k, a in algorithms.items() if k in ['adam']},
            n_iter=args.n_iter, n_points=args.n_points, cuda=args.cuda, seed=args.seed,
            id_=id_, rel_iters=True, restarts=args.restarts)

    denoised_images_linf = denoise_adversarial(
            images=images, noisy_images=noisy_images, std=args.std, dataset=args.dataset,
            norm=-1, algorithms={k: a for k, a in algorithms.items() if k in ['admm']},
            n_iter=args.n_iter, n_points=args.n_points, cuda=args.cuda, seed=args.seed,
            id_=id_, rel_iters=True, restarts=args.restarts)

    l2_adam_error = compute_error_per_checkpoint(denoised_images_l2['adam'])
    linf_al_error = compute_error_per_checkpoint(denoised_images_linf['admm'])
    baseline = np.repeat(clean_error, len(linf_al_error))
    errors_per_checkpoint = {
            r'$\ell_2$ adam': l2_adam_error,
            r'$\ell_\infty$ admm': linf_al_error,
            'base': baseline,
            }

    outf = os.path.join('draft/temp_figures/attacks/', args.dataset, id_)

    if not os.path.exists(outf):
        os.makedirs(outf)

    src.plots.iter_plot(x_axis=None,
            x_label='time', y_label='error (\%)',
            xscale='linear', yscale='linear',
            outf=outf, name='comparison_error_iter_' + str(args.std) + '.pdf', legend=True,
            smooth=10,
            variables=errors_per_checkpoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1,
            help='random seed')
    parser.add_argument('--dataset', type=str, default='mnist',
            help='dataset')
    parser.add_argument('--cuda', action='store_true',
            help='use GPU')
    parser.add_argument('--n_iter', type=int, default=4000,
            help='number of iterations for the optimization algorithm.')
    parser.add_argument('--restarts', type=int, default=1,
            help='number of restarts for the optimization algorithm.')
    parser.add_argument('--n_images', type=int, default=7,
            help='number of images to denoise.')
    parser.add_argument('--n_points', type=int,
            help='number of points along the trajectory to sample images.')
    parser.add_argument('--std', type=float, default=0.01,
            help='standard deviation for noise')

    args = parser.parse_args()
    main(args)

