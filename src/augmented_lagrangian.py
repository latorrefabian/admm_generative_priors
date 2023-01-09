import pdb
import torch

from torch import optim
from math import sqrt


class AugmentedLagrangian:
    """
    Augmented Lagrangian solver for a function fn with restriction x=g(z)

    Args:
        g (torch.Module): generative model [s] -> [d]
        fn (callable): function to be minimized. Should implement method .grad(x)
    """
    def __init__(self, fn, g):
        super().__init__()
        self.fn, self.g = fn, g

    def x_grad(self, x, z, beta, lambda_):
        """Gradient w.r.t. variable x"""
        x_grad_ = self.fn.grad(x) + lambda_ + beta * (x - self.g(z))
        return x_grad_.detach()

    def lambda_grad(self, x, z, beta, lambda_):
        """Gradient w.r.t. variable x"""
        return (x - self.g(z)).detach()

    def x_linear(self, x, z, beta, lambda_):
        """Update for x, according to the first order linearization"""
        update = (- self.fn.grad(x)
                + self.fn.L * x + beta * self.g(z) - lambda_) / (self.fn.L + beta)
        return update.detach()

    def x_exact(self, x, z, beta, lambda_):
        """Exact minimization along x variable"""
        return self.fn.x_exact(self.g, x, z, beta, lambda_)

    @staticmethod
    def adaptive_bs(beta_0, sigma_0):
        """
        Adaptive updates for beta and sigma

        Args:
            beta_0 (float): initial beta
            sigma_0 (float): inital sigma

        Return:
            callable: function that returns beta and sigma for the current iteration
        """
        beta, sigma = beta_0, sigma_0
        eta = None

        def next_(t, infs):
            """
            Compute beta and sigma for current iteration

            Args:
                t (int): iteration count
                infs: current norm of constraint tensor

            Return:
                float: current iteration beta
                float: current iteration sigma
            """
            nonlocal eta, beta, sigma

            # min_ = min(beta_0, sigma_0)

            if infs == 0:
                return beta, sigma
            if t < 2:
                return beta, sigma

            if infs <= beta_0 / sqrt(t):
            # if infs <= min_ / sqrt(t):
                next_beta = beta
            else:
                next_beta = beta * sqrt(t / (t - 1))

            if infs <= sigma_0 / t:
            # if infs <= min_ / t:
                eta = None
                next_sigma = sigma
            elif infs <= sigma_0 / sqrt(t):
            # elif infs <= min_ / sqrt(t):
                eta = None
                next_sigma = sigma * sqrt((t - 1) / t)
            else:
                if eta is None:
                    eta = sigma / (next_beta - beta)
                next_sigma = eta * (next_beta - beta)

            beta, sigma = next_beta, next_sigma

            return beta, sigma

        return next_

    def __call__(self, x, z, beta, lambda_):
        """Augmented Lagrangian value and infeasibility"""
        h = x - self.g(z)
        infs = torch.norm(h)
        al, infs = self.fn(x) + torch.sum(lambda_ * h) + beta * infs ** 2 / 2, infs
        return al, infs


def augmented_lagrangian(
        A, b, g, n_iter, lambda_0, z_0, beta=1.0, gamma=3e-5, sigma=1.0,
        x_update='linear', callback=lambda: None):
    """Augmented Lagrangian Primal-Dual Iterative Solution"""
    z = torch.tensor(z_0.data, requires_grad=True)
    x = torch.tensor(g(z).data, requires_grad=False)
    lambda_ = torch.tensor(lambda_0.data, requires_grad=False)
    aug_lagr = AugmentedLagrangian(A, b, g)
    next_bs = aug_lagr.adaptive_bs(beta_0=beta, sigma_0=sigma)
    opt = optim.SGD([z], lr=1)  # only used for .zero_grad

    for t in range(n_iter):
        if x_update == 'linear':
            x = aug_lagr.x_linear(x, z, beta, lambda_)
        elif x_update == 'exact':
            x = aug_lagr.x_exact(x, z, beta, lambda_)
        else:
            raise ValueError('x_update must be either "linear" or "exact"')

        opt.zero_grad()
        al, infs = aug_lagr(x, z, beta, lambda_)
        al.backward()
        z.data = z.data - gamma / beta * z.grad.data

        al, infs = aug_lagr(x, z, beta, lambda_)
        beta, sigma = next_bs(t, infs)
        lambda_ = lambda_ + sigma * aug_lagr.lambda_grad(x, z, beta, lambda_)

        callback(namespace=locals())

    return z


def line_search(gamma_0, theta=0.9):
    """
    Line search constructor

    Args:
       gamma_0 (float): initial step size
       theta (float): backtracking multiplier. theta \in (0, 1)
    """
    gamma = gamma_0

    def ls(fn, z):
        """
        Find the first iterate that satisfies the backtracking line search condition
        fn(x_1) <= fn(x_0) + <x_1 - x_0, fn'(x_0)> + 1 / (2 * gamma) ||x_1 - x_0||^2
        for decreasing stepsizes gamma = gamma_0 * theta^i, i >= 0.

        Args:
            fn (callable): function
            z: variable

        Return:
            torch.tensor: next iterate
        """
        nonlocal gamma

        try:
            z.grad.zero_()
        except AttributeError as e:
            pass

        fn_0 = fn(z)
        fn_0.backward()

        while True:
            z_1 = z.data - gamma * z.grad.data
            ip = torch.sum((z_1 - z) * z.grad)
            grad_ns = torch.norm(z.grad) ** 2
            fn_1 = fn(z_1)

            if fn_1 <= fn_0 + ip + gamma / 2 * grad_ns:
                gamma = gamma / theta
                break
            else:
                gamma = gamma * theta

        return z_1, torch.sqrt(grad_ns)

    return ls
