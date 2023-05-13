#!/usr/bin/env python3
r"""
Based on Pangolin
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from transformers.utils import ModelOutput

HIDDEN_SIZE = 32
# convolution window size in residual units
WINDOW_SIZE = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                21, 21, 21, 21, 41, 41, 41, 41])
# atrous rate in residual units
ATROUS_RATE = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                 10, 10, 10, 10, 25, 25, 25, 25])


class ResBlock(nn.Module):
    def __init__(self, hidden_size: int, W: int, atrous_rate: int, pad=True):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(hidden_size)
        s = 1
        # padding calculation: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338/2
        if pad:
            padding = int(1 / 2 * (1 - hidden_size + atrous_rate * (W - 1) - s + hidden_size * s))
        else:
            padding = 0
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, W, dilation=atrous_rate, padding=padding)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, W, dilation=atrous_rate, padding=padding)

    def forward(self, x):
        out = self.bn1(x)
        out = torch.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = out + x
        return out

class SpliceaiEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 4)
        self.conv = nn.Conv1d(4, hidden_size, 1)
    
    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids).permute(0, 2, 1).contiguous()
        x = self.conv(x)
        return x

class SpliceaiEncoder(nn.Module):
    def __init__(self, hidden_size, window_size: List[int]=WINDOW_SIZE, atrous_rate: List[int]=ATROUS_RATE):
        super(SpliceaiEncoder, self).__init__()
        assert len(window_size) == len(atrous_rate)
        self.window_size = window_size
        self.atrous_rate = atrous_rate
        self.skip = nn.Conv1d(hidden_size, hidden_size, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(self.window_size)):
            self.resblocks.append(ResBlock(hidden_size, self.window_size[i], self.atrous_rate[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(self.window_size))):
                self.convs.append(nn.Conv1d(hidden_size, hidden_size, 1))

    def forward(self, x: torch.Tensor):
        h = self.skip(x)
        j = 0
        for i in range(len(self.window_size)):
            x = self.resblocks[i](x)
            if (((i + 1) % 4 == 0) or ((i + 1) == len(self.window_size))):
                dense = self.convs[j](x)
                j += 1
                h = h + dense
        return ModelOutput(last_hidden_state=h)

class SpliceaiModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=HIDDEN_SIZE, window_size: List[int]=WINDOW_SIZE, atrous_rate: List[int]=ATROUS_RATE):
        super(SpliceaiModel, self).__init__()
        self.embedding = SpliceaiEmbedding(vocab_size, hidden_size)
        self.encoder = SpliceaiEncoder(hidden_size, window_size, atrous_rate)
    
    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids)
        sequence_output = self.encoder(x).last_hidden_state
        return ModelOutput(last_hidden_state=sequence_output)

class SpliceaiForMaskedLM(nn.Module):
    def __init__(self, vocab_size, hidden_size: int=HIDDEN_SIZE, window_size: List[int]=WINDOW_SIZE, atrous_rate: List[int]=ATROUS_RATE):
        super().__init__()
        self.model = SpliceaiModel(vocab_size, hidden_size, window_size, atrous_rate)
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> ModelOutput:
        outputs = self.model(input_ids)
        sequence_output = outputs.last_hidden_state.permute(0, 2, 1)
        logits = self.classifier(sequence_output)
        return ModelOutput(logits=logits)

class SpliceaiForTokenClassification(nn.Module):
    def __init__(self, vocab_size, num_classes, hidden_size: int=32, window_size: List[int]=WINDOW_SIZE, atrous_rate: List[int]=ATROUS_RATE):
        super().__init__()
        self.model = SpliceaiModel(hidden_size, window_size, atrous_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> ModelOutput:
        outputs = self.model(input_ids)
        sequence_output = outputs.last_hidden_state.permute(0, 2, 1)
        logits = self.classifier(sequence_output)
        return ModelOutput(logits=logits)
