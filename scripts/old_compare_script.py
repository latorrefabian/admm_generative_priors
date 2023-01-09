import pdb
import sys, os, datetime, platform
sys.path.insert(0, './')
import torch
import torchvision.utils as tv_utils
import src.plots

from src.functions import measurement_error, squared_error
from src.utils import random_z, Info, IterPrinter, Callback
from src.models.celeb import DCGAN_G, DCGAN_G_ELU
from src.models.mnist import MNIST_FC_ELU, MNIST_FC_G

from timeit import default_timer as timer

from src import algorithms
from src.augmented_lagrangian2 import augmented_lagrangian, augmented_lagrangian4, augmented_lagrangian5
from src.new_augmented_lagrangian import augmented_lagrangian as new_augmented_lagrangian


manual_seed = 20
torch.manual_seed(manual_seed)
outf = 'document/figures'
nz, ngf, nc = 100, 64, 3
g_path = 'pretrained_generators/celeba_dcgan.pt'
noise_shape = [1, nz, 1, 1]
m = 1000
n_iter = 1000000

mnist = True
id_ = False

if mnist:
    nz, ngf, nc = 20, 200, 1
    generator = MNIST_FC_ELU(nz=nz, ngf=ngf, nc=nc)
else:
    nz, ngf, nc = 100, 64, 3
    g_path = 'pretrained_generators/celeba_dcgan.pt'
    generator = DCGAN_G_ELU(ngpu=1, nz=nz, ngf=ngf, nc=nc)
    generator.load_state_dict(
            torch.load(g_path, map_location='cpu'))

noise_shape = [1, nz, 1, 1]

n_iter = 2000
m = 200

for param in generator.parameters():
    param.requires_grad = False

generator.eval()

if platform.system() == 'Darwin':
    cuda = False
else:
    cuda = True

z_true = random_z(distribution='uniform', shape=noise_shape)
x_true = generator(z_true)
z_0 = random_z(distribution='uniform', shape=z_true.shape)
x_0 = generator(z_0)
lambda_0 = torch.zeros_like(x_true)
A = torch.zeros(m, x_true.numel()).normal_()
b = A @ x_true.view(-1)

if cuda:
    z_true = z_true.cuda()
    z_0 = z_0.cuda()
    x_0 = x_0.cuda()
    x_true = x_true.cuda()
    lambda_0 = lambda_0.cuda()
    A = A.cuda()
    b = b.cuda()
    generator.cuda()

me = measurement_error(A, b)
se = squared_error(x_true)

params = {
    'gal1': {
        'lr': 0.0001,
        'lambda_0': lambda_0,
        'beta': 0.1,
        'sigma': 0.01,
        'adaptive': 1,
    },
    'gpgd': {
        'lr': 0.0000003,
        'n_iter_p': 100,
        'lr_x': 0.0005,
        },
    'ggd': {
        'lr': 0.0001,
        },
}

variables = {
    'gal1': ['al', 'ref_value', 'fn_value', 'infeasibility', 'beta', 'sigma', 'grad_norm'],
    'al2': ['fn_value', 'ref_value', 'ifs', 'beta', 'sigma', 'grad_norm'],
    'gpgd': ['ref_value', 'fn_value', 'grad_norm'],
    'ggd': ['ref_value', 'fn_value', 'grad_norm'],
}

algorithms = {
        'ggd': algorithms.ggd,
#        'gal1': algorithms.gal,
#        'al2': augmented_lagrangian,
#        'gpgd': algorithms.gpgd,
}

images = dict()
fn_values = dict()
ref_values = dict()
grad_norm = dict()
ifs = dict()
beta_val = dict()
sigma_val = dict()

for k, algorithm in algorithms.items():
    if k == 'al2':
        continue
    info = Info(*variables[k])
    iter_print = IterPrinter(*variables[k])
    print(k)
    start = timer()
    images[k] = algorithm(
            fn=me, g=generator, z_0=z_0, n_iter=n_iter,
            ref=se, info=info, iter_print=iter_print, **params[k])
    end = timer()
    print('time:', end - start)
    fn_values[k] = info['fn_value']
    # ref_values[k] = info['ref_value']
    grad_norm[k] = info['grad_norm']

info = Info(*variables['al2'])
iter_print = IterPrinter(*variables['al2'])

callback = Callback(
        fn_z='torch.norm(A @ g(z).view(-1) - b) ** 2',
        fn_x='torch.norm(A @ x.view(-1) - b) ** 2',
        infs='infs', grad_norm='grad_norm', beta='beta', sigma='sigma')

# print('augmented lagrangian 5')
# start = timer()
# result_new = augmented_lagrangian5(
#         A=A, b=b, g=generator, n_iter=n_iter, lambda_0=lambda_0, z_0=z_0, beta=1, gamma=.2, sigma=4,
#         callback=callback)
# end = timer()
# print('time:', end - start)
# 
# fn_values['al'] = callback['fn_z']
# ref_values['al'] = callback['fn_x']
# ifs['al'] = callback['infs']
# callback.clear()

print('new augmented lagrangian')
start = timer()
result = new_augmented_lagrangian(
        A=A, b=b, g=generator, n_iter=n_iter, lambda_0=lambda_0, z_0=z_0, beta=100, gamma=0.1, sigma=100,
        callback=callback)
end = timer()
print('time:', end - start)

fn_values['al_new'] = callback['fn_z']
ref_values['al_new'] = callback['fn_x']
ifs['al_new'] = callback['infs']
beta_val['al_new'] = callback['beta']
sigma_val['al_new'] = callback['sigma']

id_ = datetime.datetime.now().strftime('%I%M%p_%B_%d_%Y')
outf = os.path.join(outf, id_)
if not os.path.exists(outf):
    os.makedirs(outf)

src.plots.iter_plot(
        x_label='iterations',
        y_label=r'$||Ag(z) - b||^2$', xscale='log', yscale='log', outf=outf, name='z_measurement_error.pdf',
        **fn_values)

src.plots.iter_plot(
        x_label='iterations',
        y_label=r'$||Ax - b||^2$', xscale='log', yscale='log', outf=outf, name='x_measurement_error.pdf',
        **ref_values)

src.plots.iter_plot(
        x_label='iterations',
        y_label=r'$||x - g(z)||$', xscale='log', yscale='log', outf=outf, name='infeasibility.pdf',
        **ifs)

# src.plots.iter_plot(
#         x_label='iterations',
#         y_label=r'$||\nabla f||^2$', xscale='log', yscale='log', outf=outf, name='grad_norm.pdf',
#         **grad_norm)

src.plots.iter_plot(
        x_label='iterations',
        y_label=r'$\beta$', xscale='log', yscale='log', outf=outf, name='beta.pdf',
        **beta_val)

src.plots.iter_plot(
        x_label='iterations',
        y_label=r'$\sigma$', xscale='log', yscale='log', outf=outf, name='sigma.pdf',
        **sigma_val)

image_list = []
filename = []

for k, im in images.items():
    filename.append(k)
    image_list.append(im[0])

tv_utils.save_image(
        [x_0[0]] + image_list + [x_true[0]],
        os.path.join(outf, '_'.join(filename) + '.png'), normalize=True)
