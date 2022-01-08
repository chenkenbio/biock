#!/usr/bin/env python3

import torch
import torch.nn as nn

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

