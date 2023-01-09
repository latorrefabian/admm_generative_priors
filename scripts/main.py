import pdb
import argparse, random, os
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as tv_utils
import numpy as np
import src.plots

from collections import defaultdict

from src.models.celeb import DCGAN_G, FC_G
from src import algorithms
from src.utils import Info, compare_recovery


gen_params = {
    'dcgan': {
        'ngpu': 1,
        'nz': 100,
        'ngf': 64,
        'nc':3
        },
    'fc': {
        'input_dim': 100,
        'output_dim': np.array([3, 64, 64])
        }
    }

z_dim = {
    'dcgan': (1, 100, 1, 1),
    'fc': (1, 100)
    }

gen_path = defaultdict(lambda: None)
gen_path.update(
    dcgan='pretrained_generators/celeba_dcgan.pt'
    )

def load_generator(name):
    if name == 'dcgan':
        Generator = DCGAN_G
    elif name == 'fc':
        Generator = FC_G
    else:
        raise ValueError

    generator = Generator(**gen_params[name])
    map_location = None if args.cuda else 'cpu'  # map tensors to cpu

    if gen_path[name] is not None:
        generator.load_state_dict(
                torch.load(gen_path[name], map_location=map_location))

    for param in generator.parameters():
        param.requires_grad = False

    generator.eval()

    return generator


def main(args):
    generator = load_generator(name=args.generator)

    z_true = torch.FloatTensor(*z_dim[args.generator]).uniform_(-1, 1)
    x_true = generator(z_true)

    z_0 = torch.zeros_like(z_true).uniform_(-1, 1)
    x_0 = generator(z_0)

    pgd = algorithms.pgd
    n_measurements = [2 ** k for k in range(5, 12)]

    results = compare_recovery(
            x_true=x_true[0], g=generator, z_0=z_0,
            n_measurements=n_measurements, pgd=pgd)

    src.plots.iter_plot(
            x_axis=n_measurements, x_label='\# measurements',
            y_label=r'$\frac{1}{d}||x - x^\star||_2$', **results['recovery'])

    src.plots.iter_plot(
            x_label='\# iterations',
            y_label=r'$\frac{1}{m}||Ax - b||^2$', **results['optimization'])

    tv_utils.save_image(
            [x_0[0]] + results['images']['pgd'] + [x_true[0]],
            os.path.join(args.outf, 'recovered.png'), normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator', default='dcgan', help='dcgan | fc')
    parser.add_argument('--z_distribution', default='uniform', help='uniform | normal')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--g_path', default='pretrained_generators/celeba_dcgan.pt', help='path to generator checkpoint')
    parser.add_argument('--outf', default='document/figures', help='folder to output images and plots')
    parser.add_argument('--seed', type=int, help='manual seed')
    parser.add_argument('--profile', action='store_true', help='enable cProfile')
    args = parser.parse_args()

    print('**** Arguments ****')
    for arg, value in args._get_kwargs():
        print(arg + ':', value)
    print('*******************')

    # process arguments
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run "
              "with --cuda")

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print('Random Seed:', args.seed)
    print('*******************')

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = True  # turn on the cudnn autotuner

    main(args)
