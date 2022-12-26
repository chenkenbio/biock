#!/usr/bin/env python3
"""
Author: Ken Chen
Email: chenkenbio@gmail.com
Date: 2022-12-25
"""

import argparse
import os
import sys
import numpy as np
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union, Set
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

class SpanMasker():
    def __init__(self, vocab_size: List[int], mask_id, num_special_token: int, unknown_token_id: int, pad_id: int=-100, min_span=1, max_span: int=10, p: float=0.2, mask_rate: float=0.15, with_special_token: bool=True) -> None:
        r"""
        unknown_token_id: N/n
        """
        self.vocab_size = vocab_size
        self.unknown_token_id = unknown_token_id
        self.with_special_token = with_special_token
        self.num_special_token = num_special_token
        self.shift = 1 if with_special_token else 0
        self.mask_id = mask_id
        self.pad_id = pad_id
        self.lower = min_span
        self.upper = max_span
        self.mask_id = mask_id
        self.mask_rate = mask_rate
        self.p = p
        self.lens = list(range(self.lower, self.upper + 1))
        self.len_dist = [self.p * (1 - self.p) ** (i - self.lower) for i in range(self.lower, self.upper + 1)]
        self.len_dist = [x / (sum(self.len_dist)) for x in self.len_dist]
    
    def mask(self, input_ids: np.ndarray):
        assert input_ids.ndim == 1
        seq_len = input_ids.shape[0] - 2 if self.with_special_token else input_ids.shape[0]
        input_ids = np.copy(input_ids)
        mask = set()
        mask_num = np.ceil(seq_len * self.mask_rate)
        spans = list()
        while len(mask) < mask_num:
            span_len = np.random.choice(self.lens, p=self.len_dist)
            anchor = np.random.choice(seq_len)
            if anchor in mask:
                continue
            left, right = anchor, anchor
            for i in range(anchor, min(seq_len, anchor + span_len)):
                p = i + self.shift
                if len(mask) == mask_num:
                    break
                if p in mask:
                    continue
                mask.add(p)
                right = p
            if left < right:
                spans.append([left + 1, right + 1])
        
        spans = merge_intervals(spans)
        target = np.full(input_ids.shape, fill_value=self.pad_id)
        for start, end in spans:
            rand = np.random.random()
            for i in range(start, end):
                if input_ids[i] == self.unknown_token_id:
                    continue
                target[i] = input_ids[i]
                if rand < 0.8:
                    input_ids[i] = self.mask_id
                elif rand < 0.9:
                    input_ids[i] = np.random.randint(low=self.num_special_token, high=self.vocab_size)
        return input_ids, target


def merge_intervals(intervals):
    intervals = sorted(intervals, key=lambda x : x[0])
    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] + 1 < interval[0]:
            merged.append(interval)
        else:
        # otherwise, there is overlap, so we merge the current and previous
        # intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged


from sklearn.metrics import accuracy_score

def calculate_mlm_acc(input_ids: Tensor, target: Tensor, prediction: Tensor, mask: Tensor=None) -> Tuple[float, float]:
    r"""
    Return
    ---
    MLMACC : accuracy of masked tokens
    ALLACC : accuracy of all tokens
    """
    assert input_ids.shape == target.shape, "{}".format((input_ids.shape, target.shape, prediction.shape))
    assert prediction.ndim == input_ids.ndim or prediction.ndim - 1 == input_ids.ndim, "{}".format((input_ids.shape, target.shape, prediction.shape))
    if prediction.ndim - 1 == input_ids.ndim:
        prediction = torch.argmax(prediction, dim=-1)
    input_ids = input_ids.reshape(-1).numpy()
    target = target.reshape(-1).numpy()
    prediction = prediction.reshape(-1).numpy()
    mask = mask.reshape(-1).numpy()
    keep = np.where(mask > 0)[0]
    input_ids = input_ids[keep]
    target = target[keep]
    prediction = prediction[keep]
    mask = (target != -100)
    mlm_acc = accuracy_score(target[mask], prediction[mask])
    unmask_acc = accuracy_score(input_ids[~mask], prediction[~mask])
    return mlm_acc, unmask_acc
