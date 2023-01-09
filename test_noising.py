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
from src.utils import Callback
from src.params import params
from napalm import algorithms as alg

from torch import optim

import torchvision.utils as tv_utils

from src.experiments import load_test_images, denoise_adversarial
from src.models.loader import load_classifier
from src.util import attack

dim_space = {
        'mnist': 28 * 28,
        'celeba': 3 * 64 * 64,
        'lsun': 3 * 64 * 64,
        }

ckpt_mnist = 'pretrained_classifier/mnist_0_1/300-300-300-0.0.ckpt'
#ckpt_mnist = 'pretrained_classifier/mnist/300-300-300-robust.ckpt'

def main(args):
    torch.manual_seed(args.seed)
    algorithms = {
                'gd': alg.gd, 'admm': alg.admm,
                'al': alg.al, 'adam': alg.adam
    }

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

    adv_noisy_images = pgm.attack(
            classifier,
            optimizer=optimizer,
            data_batch=images, label_batch=labels)

    adv_noisy_images = torch.clamp(adv_noisy_images, 0, 1).detach()
    print('error on attacked images: ', compute_error(adv_noisy_images), '%')

    noisy_images = torch.clamp(images + args.std * torch.sign(torch.zeros_like(images).normal_()), 0, 1)
    print('error on noisy images: ', compute_error(noisy_images), '%')

    noisy_adv_images = torch.clamp(adv_noisy_images + args.std * torch.sign(torch.zeros_like(images).normal_()), 0, 1)
    print('error on noisy attacked images: ', compute_error(noisy_adv_images), '%')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1,
            help='random seed')
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
    parser.add_argument('--std', type=float, default=0.1,
            help='standard deviation for noise')

    args = parser.parse_args()
    main(args)
