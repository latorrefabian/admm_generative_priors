import torch
from itertools import islice, chain

from napalm import algorithms as alg
from napalm import functions as nap_fn
from napalm import utils

from src.util import attack_2 as attack
from src.models.loader import load_classifier, load_generator
from torchvision import datasets, transforms

ckpt_mnist = 'pretrained_classifier/mnist_0_1/300-300-300-0.0.ckpt'
ckpt_g_mnist = 'pretrained_generators/elu/mnist_netG.ckpt'

def main():
    cl = load_classifier(model2load=ckpt_mnist, dataset='mnist')
    g = load_generator(ckpt_g_mnist, dataset='mnist', input_dim=128, elu=True)
    cl.eval()
    g.eval()
    for p in chain(cl.parameters(), g.parameters()):
        p.requires_grad = False

    perturb = attack.pgm(
            classifier=cl, n_iters=30, epsilon=.2, p=float('inf'))
    test_error = attack.test_error(g, transform=perturb)
    dataset = datasets.MNIST(
        root='~/.data', train=False, download=True,
        transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=100, shuffle=True)
    error, adv_error = test_error(islice(test_loader, 0, 100))
    print('error: ' + str(error) + '%')
    print('adv_error: ' + str(adv_error) + '%')


def main2():
    cl = load_classifier(model2load=ckpt_mnist, dataset='mnist')
    g = load_generator(ckpt_g_mnist, dataset='mnist', input_dim=128, elu=True)
    cl.eval()
    g.eval()
    for p in chain(cl.parameters(), g.parameters()):
        p.requires_grad = False

    perturb = attack.pgm(
            classifier=cl, n_iters=30, epsilon=.2, p=float('inf'))
    dataset = datasets.MNIST(
        root='~/.data', train=False, download=True,
        transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=100, shuffle=True)

    denoised = []
    for x, y in islice(test_loader, 0, 2):
        denoised.append(
            denoise_batch_linf(g, perturb, x, y)
        )
            

def denoise_batch_linf(g, perturb, x, y):
    x_adv, _ = perturb(x, y)
    z_0 = torch.zeros(128, device=x.device)
    denoised = torch.zeros_like(x_adv)
    dim_image = x_adv[0].numel()

    def fn(x, z):
        return 0.0

    def h(x, z):
        return x - g(z)

    for i in range(x_adv.shape[0]):
        noisy_image = x_adv[i]

        def prox_R(x, lambda_):
            nap_fn.prox_linf(x, lambda_) + noisy_image

        callback = utils.Callback(
            resources=dict(dim=dim_image),
            variables=dict(
                    fn_val='fn_value.item()',
                    fn_best='fn_best.item()',
                    infs='infs.item()/dim',
                )
            )

        z_best = alg.admm(
            fn=fn, h=h, x_0=g(z_0), z_0=z_0, n_iter=100, L=0,
            beta=1e-6, sigma=1e-6, gamma_z=0.015, callback=callback)['z_best']
        denoised[i] = z_best

    return denoised


if __name__ == '__main__':
    main2()

