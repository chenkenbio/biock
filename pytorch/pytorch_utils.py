#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import Literal, Optional

## genome one hot
# 'N': 0
# 'A': 1
# 'C': 2
# 'G': 3
# 'T': 4
ONEHOT = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)


def model_summary(model):
    """
    model: pytorch model
    """
    total_param = 0
    trainable_param = 0
    for i, p in enumerate(model.parameters()):
        num_p = torch.numel(p)
        if p.requires_grad:
            trainable_param += num_p
        total_param += num_p
    return {'total_param': total_param, 'trainable_param': trainable_param}


def make_onehot(N: int, include_zero: bool=False, dtype=torch.float) -> Tensor:
    onehot = torch.as_tensor(np.diag(np.ones(N)), dtype=dtype)
    if include_zero:
        onehot = torch.cat((torch.zeros((1, N)), onehot), dim=0),
    return onehot


def kl_divergence(mu1: Tensor, logvar1: Tensor, mu2: Optional[Tensor]=None, logvar2: Optional[Tensor]=None, reduction: Literal["mean"]="mean") -> Tensor:
    if mu2 is None:
        mu2 = torch.zeros_like(mu1)
        logvar2 = torch.zeros_like(logvar1)
    kl_div = 0.5 * (logvar2 - logvar1 + (logvar1.exp() + (mu1 - mu2).pow(2)) / logvar2.exp() - 1)
    kl_div = kl_div.mean(dim=1)
    if reduction == "mean":
        kl_div = torch.mean(kl_div)
    return kl_div


def set_seed(seed: int):
    if float(torch.version.cuda) >= 10.2:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    
