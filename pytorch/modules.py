#!/usr/bin/env python3
"""
Author: Ken Chen (chenkenbio@gmail.com)
Date: <<date>>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable

class SelfAttentionEmbedding(nn.Module):
    """
    based on https://github.com/kaushalshetty/Structured-Self-Attention
    """
    def __init__(self, d_model: int, da: int, r: int, require_penalty: bool=False, skip_max: bool=False):
        super(SelfAttentionEmbedding, self).__init__()
        self.da = da
        self.r = r
        self.att_first = nn.Linear(d_model, da)
        self.att_first.bias.data.fill_(0)
        self.att_second = nn.Linear(da, r)
        self.att_second.bias.data.fill_(0)
        self.require_penalty = require_penalty
        self.skip_max = skip_max
    
    def forward(self, seq: Tensor) -> Tensor:
        ## input: (B, S, E)
        bs, device  = seq.size(0), seq.device
        att = torch.tanh(self.att_first(seq))  #(B, S, E) -> (B, S, da)
        att = F.softmax(self.att_second(att), dim=1).transpose(1, 2) # (B, S, da) -> (B, S, r) -> (B, r, S)
        seq = torch.matmul(att, seq) # (B, r, S) x (B, S, E) -> (B, r, E)
        if self.skip_max:
            seq = seq.mean(dim=1).view(bs, -1)
        else:
            seq = torch.cat((
                seq.mean(dim=1).view(bs, -1),
                seq.max(dim=1)[0].view(bs, -1)
            ), dim=1)
        if self.require_penalty:
            identity = torch.eye(self.r).to(device) # (r, r)
            identity = Variable(identity.unsqueeze(0).expand(bs, self.r, self.r)) # (B, r, r)
            penal = torch.linalg.matrix_norm(torch.matmul(att, att.transpose(1, 2)) - identity).mean() # different from the original implementation
            return seq, penal
        else:
            return seq
