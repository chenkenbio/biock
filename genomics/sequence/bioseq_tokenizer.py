#!/usr/bin/env python3
"""
Author: Ken Chen
Email: chenkenbio@gmail.com
Date: 2022-11-19
"""

import argparse
import os
import sys
import numpy as np
import collections
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
import logging
logger = logging.getLogger(__name__)

from ._sequence_to_kmer_list import _fast_seq_to_kmer_list
from ._encode_sequence import _encode_sequence

def fast_seq_to_kmer_list(text: str, k=1, pad: bool=False, shift: int=None) -> List[str]:
    if shift is None:
        shift = 1
    if k == 1:
        assert shift == 1
        return ' '.join(list(text))
    else:
        if pad:
            return _fast_seq_to_kmer_list(text, k, k//2, k, shift)
        else:
            return _fast_seq_to_kmer_list(text, k, 0, k, shift)

def encode_sequence(seq: str, token_dict: Dict[str, int]=None) -> List[int]:
    if token_dict is None:
        ids = _encode_sequence(seq)
    else:
        ids = _encode_sequence(seq, token_dict)
    return ids
    


def seq_to_kmer_list(seq, k):
    pad = k // 2
    end = pad + len(seq)
    kmers = list()
    pad_seq = ''.join(['N' * pad, seq, 'N' * pad])

    for i in range(pad, end):
        start = i - pad
        kmers.append(pad_seq[start:start + k])
    return kmers

_NN2INDEX = {n: i for i, n in enumerate(list("NACGT"))}
_NN2INDEX['U'] = 4
for n in "NACGTU":
    _NN2INDEX[n.lower()] = _NN2INDEX[n]

_ONEHOT = np.concatenate((np.ones((1, 4)), np.diag(np.ones(4))))
def onehot_dna(seq):
    seq = np.asarray([_NN2INDEX[n] for n in seq])
    return _ONEHOT[seq]