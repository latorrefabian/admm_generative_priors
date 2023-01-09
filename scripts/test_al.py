import pdb
import sys, os, datetime, platform
sys.path.insert(0, './')
import torch
import torchvision.utils as tv_utils
import src.plots

from src.functions import measurement_error, generator_mismatch
from src.utils import load_generator, random_z, run_al, Info, IterPrinter
from src.models.celeb import DCGAN_G, DCGAN_G_ELU
from src.augmented_lagrangian import augmented_lagrangian


manual_seed = 7
torch.manual_seed(manual_seed)
outf = 'document/figures'
nz, ngf, nc = 100, 64, 3
g_path = 'pretrained_generators/celeba_dcgan.pt'
bs = 1
noise_shape = [bs, nz, 1, 1]
m = 1000
beta = 0.001
sigma = 10 * beta
n_iter = 1000000
gamma = 2e-5

if platform.system() == 'Darwin':
    cuda = False
else:
    cuda = True

print('cuda:' + str(cuda))
load = False

generator = DCGAN_G_ELU(ngpu=1, nz=nz, ngf=ngf, nc=nc)
generator.load_state_dict(
            torch.load(g_path, map_location='cpu'))

for param in generator.parameters():
    param.requires_grad = False

generator.eval()

if load:
    x_load = torch.load('x_cpu.pt')
    z_0 = x_load['z'].data
    x_0 = x_load['x'].data
    lambda_0 = x_load['lambda_']
    A = x_load['A']
    b = x_load['b']
else:
    z_true = random_z(distribution='uniform', shape=noise_shape)
    z_0 = random_z(distribution='uniform', shape=z_true.shape)
    x_0 = generator(z_0)
    x_true = generator(z_true)
    lambda_0 = torch.zeros_like(x_true)
    A = torch.zeros(m, x_true[0].numel()).normal_()
    b = torch.matmul(x_true.view(bs, -1), A.t())

if cuda:
    z_true = z_true.cuda()
    z_0 = z_0.cuda()
    x_0 = x_0.cuda()
    x_true = x_true.cuda()
    lambda_0 = lambda_0.cuda()
    A = A.cuda()
    b = b.cuda()
    generator.cuda()

bs = x_true.size(0)
me = measurement_error(A, b)
gm = generator_mismatch(g=generator)

info = Info('al', 'fn_value', 'infeasibility', 'beta', 'sigma', 'grad_norm')
iter_print = IterPrinter('al', 'fn_value', 'infeasibility', 'beta', 'sigma', 'grad_norm')

def fn_(z, **kwargs):
    return me(generator(z))

results = augmented_lagrangian(
        fn=me, h=gm, n_iter=n_iter, lambda_0=lambda_0,
        beta=beta, gamma=gamma, sigma=sigma,
        info=info, iter_print=iter_print, fn_=fn_, x_0=x_0, z_0=z_0)

recovered_x = generator(results['z'])

id_ = datetime.datetime.now().strftime('%I%M%p_%B_%d_%Y')
outf = os.path.join(outf, id_)
if not os.path.exists(outf):
    os.makedirs(outf)

src.plots.iter_plot(
        x_label='iterations',
        y_label=r'Augmented Lagrangian', xscale='log', yscale='symlog', outf=outf, name='aug_lagr.pdf',
        augmented_lagrangian=info['al'])

src.plots.iter_plot(
        x_label='iterations',
        y_label=r'$||Ax - b||^2$', xscale='log', yscale='log', outf=outf, name='measurement_error.pdf',
        measurement_error=info['fn_value'])

src.plots.iter_plot(
        x_label='iterations',
        y_label=r'$||G(z) - w||$', xscale='log', yscale='log', outf=outf, name='infeasibility.pdf',
        infeasibility=info['infeasibility'][2:])

src.plots.iter_plot(
        x_label='iterations',
        y_label=r'$\sigma$', xscale='log', yscale='log', outf=outf, name='beta.pdf',
        beta=info['beta'])

src.plots.iter_plot(
        x_label='iterations',
        y_label=r'$\sigma$', xscale='log', yscale='log', outf=outf, name='sigma.pdf',
        sigma=info['sigma'])

src.plots.iter_plot(
        x_label='iterations',
        y_label=r'$||\nabla \mathcal{L}||$', xscale='log', yscale='log', outf=outf, name='grad_norm.pdf',
        grad_norm=info['grad_norm'])

tv_utils.save_image(
        [x_0[0]] + [recovered_x[0]] + [x_true[0]],
        os.path.join(outf, 'al_recovered.png'), normalize=True)
