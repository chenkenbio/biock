#!/usr/bin/env python3

import argparse
import math
import numpy as np
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
import torch.nn.functional as F
from .pytorch_utils import kl_divergence
from .autoencoders import build_mlp, Decoder, NBDecoder, ZINBDecoder, PoissonDecoder

import logging
logger = logging.getLogger(__name__)

# def kl_divergence(mu, logvar):
#     """
#         Computes the KL-divergence of
#         some element z.
# 
#         KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
#                  = -E[log p(z) - log q(z)]
#     """
#     return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def binary_cross_entropy(recon_x, x):
    return -torch.sum(x * torch.log(recon_x + 1e-8) + (1 - x) * torch.log(1 - recon_x + 1e-8), dim=-1)


def elbo(recon_x, x, z_params, binary=True):
    """
    elbo = likelihood - kl_divergence
    L = -elbo

    Params:
        recon_x:
        x:
    """
    mu, logvar = z_params
    kld = kl_divergence(mu, logvar)
    if binary:
        likelihood = -binary_cross_entropy(recon_x, x)
    else:
        # likelihood = -F.mse_loss(recon_x, x)
        likelihood = -F.smooth_l1_loss(recon_x, x, beta=10)
        # likelihood = - ((recon_x - x) * x / x.size(1) * (recon_x - x)).sum(dim=1)
    return torch.sum(likelihood), torch.sum(kld)
    # return likelihood, kld

class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / var
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar

class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """
    def reparametrize(self, mu, logvar):
        epsilon = torch.randn(mu.size(), requires_grad=False, device=mu.device)
        std = logvar.mul(0.5).exp_()
#         std = torch.clamp(logvar.mul(0.5).exp_(), -5, 5)
        z = mu.addcmul(std, epsilon)

        return z

class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """
    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)
        return self.reparametrize(mu, log_var), mu, log_var

class VAEEncoder(nn.Module):
    def __init__(self, x_dim: int, h_dim: List[int], z_dim: int, bn=True, dropout=0):
        """
        Inference network

        Attempts to infer the probability distribution
        p(z|x) from the data by fitting a variational
        distribution q_φ(z|x). Returns the two parameters
        of the distribution (µ, log σ²).

        :param dims: dimensions of the networks
           given by the number of neurons on the form
           [input_dim, [hidden_dims], latent_dim].
        """
        super(VAEEncoder, self).__init__()
        self.hidden = build_mlp([x_dim]+h_dim, bn=bn, dropout=dropout)
        self.sample = GaussianSample(([x_dim]+h_dim)[-1], z_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Args:
            x: input tensor

        Return:
            z: Tensor
            mu: Tensor
            logvar: Tensor
        """
        x = self.hidden(x)
        return self.sample(x) ## -> z, mu, log_var



def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #p.add_argument()
    p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
