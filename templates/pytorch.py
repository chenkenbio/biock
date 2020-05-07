#!/usr/bin/env python3

from importlib import reload

## import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.autograd import Variable


## data operation
import numpy as np
import pandas as pd


## deterministic
seed = 2020
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(seed)


## model summary
def model_summary(model):
    total_param = 0
    trainable_param = 0
    for p in model.parameters():
        num_p = len(p)
        if p.requires_grad:
            trainable_param += num_p
        total_param += num_p
    return {'total_param': total_param, 'trainable_param': trainable_param}
