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
import logging

logger = logging.getLogger(__name__)


def build_mlp(layers, nhead: int=1, activation=nn.ReLU(), bn=True, dropout=0):
    """
    Build multilayer linear perceptron
    """
    net = []
    for i in range(1, len(layers)):
        if nhead == 1:
            net.append(nn.Linear(layers[i - 1], layers[i]))
        else:
            net.append(MultiHeadLinear(layers[i - 1], layers[i], nhead=nhead))
        if bn:
            net.append(nn.BatchNorm1d(layers[i]))
        net.append(activation)
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


class AEEncoder(nn.Module):
    def __init__(self, x_dim: int, h_dim: List[int], z_dim: int, bn=True, dropout=0):
        super(AEEncoder, self).__init__()
        self.hidden = build_mlp([x_dim]+h_dim, bn=bn, dropout=dropout)
        self.sample = nn.Sequential(
            nn.Linear(([x_dim] + h_dim)[-1], z_dim),
            nn.LeakyReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x: input tensor

        Return:
            z
        """
        x = self.hidden(x)
        return self.sample(x) ## -> z


class Decoder(nn.Module):
    def __init__(self, x_dim: int, h_dim: List[int], z_dim: int, output_activation, bn=True, dropout=0, eps=0):
        """
        basic decoder, single output, range
        """
        super(Decoder, self).__init__()
        self.eps = eps

        self.hidden = build_mlp([z_dim, *h_dim], bn=bn, dropout=dropout)
        self.reconstruction = nn.Linear([z_dim, *h_dim][-1], x_dim)

        self.output_activation = output_activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.hidden(x)
        x = self.reconstruction(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x + self.eps


class NBDecoder(nn.Module):
    def __init__(self, x_dim: int, h_dim: List[int], z_dim: int, bn=True, dropout=0, output_activation=nn.Softplus(), eps=1E-8):
        r"""
        Return
        -------
        mu (>0)
        theta (>0)
        """
        super(NBDecoder, self).__init__()

        self.hidden = build_mlp([z_dim, *h_dim], bn=bn, dropout=dropout)
        self.eps = eps
#         self.hidden = build_mlp([z_dim]+h_dim, bn=bn, dropout=dropout)
        self.mu = nn.Sequential(
            nn.Linear([z_dim, *h_dim][-1], x_dim),
            output_activation
        )
        self.theta = nn.Sequential(
            nn.Linear([z_dim, *h_dim][-1], x_dim),
            output_activation
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.hidden(x)
        return self.mu(x) + self.eps, self.theta(x) + self.eps

class ZINBDecoder(NBDecoder):
    def __init__(self, x_dim: int, h_dim: List[int], z_dim: int, bn=True, dropout=0, output_activation=nn.Softplus(), eps=1E-8):
        super().__init__(x_dim, h_dim, z_dim, bn, dropout, output_activation, eps=eps)
        self.zi_logits = nn.Linear([z_dim, *h_dim][-1], x_dim)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.hidden(x)
        return self.mu(x) + self.eps, self.theta(x) + self.eps, self.zi_logits(x)
 

class PoissonDecoder(Decoder):
    def __init__(self, x_dim: int, h_dim: List[int], z_dim: int, output_activation=nn.Softplus(), bn=True, dropout=0, eps=1E-8):
        super().__init__(x_dim, h_dim, z_dim, output_activation, bn, dropout)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x = self.reconstruction(self.hidden(x))
        x = self.output_activation(x)
        return x + self.eps

        
class MultiHeadLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, nhead: int, bias: bool=True, device=None, dtype=None) -> None:
        super(MultiHeadLinear, self).__init__()
        assert nhead > 1
        self.in_features = in_features
        self.out_features = out_features
        self.nhead = nhead
        self.weight = nn.Parameter(torch.empty(nhead, in_features, out_features)) # (D, H, H')
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, nhead))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        ## input: (B, H) or (B, H, D)
        # return F.linear(input, self.weight, self.bias)
        assert len(input.size()) <= 3
        if len(input.size()) == 3:
            assert input.size(2) == self.weight.size(0), "dimension should be same in MultiHeadLinear, while input: {}, weight: {}".format(input.size(), self.weight.size())
            input = input.transpose(0, 1).transpose(0, 2)
        input = torch.matmul(input, self.weight).transpose(0, 1).transpose(1, 2)
        if self.bias is not None:
            input += self.bias
        return input

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, nhead={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.nhead
        )


