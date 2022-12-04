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

from _sequence_to_kmer_list import _fast_seq_to_kmer_list

def fast_seq_to_kmer_list(text: str, k=1, pad: bool=False, shift: int=None):
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


def seq_to_kmer_list(seq, k):
    pad = k // 2
    end = pad + len(seq)
    kmers = list()
    pad_seq = ''.join(['N' * pad, seq, 'N' * pad])

    for i in range(pad, end):
        start = i - pad
        kmers.append(pad_seq[start:start + k])
    return kmers
