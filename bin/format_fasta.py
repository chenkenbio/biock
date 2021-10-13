#!/usr/bin/env python3

import argparse, os, sys, time
# import logging, warnings, json, gzip, pickle
# warning = logging.warning
from biock.biock import custom_open


#from collections import defaultdict, OrderedDict

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from scipy.stats import pearsonr, spearmanr
from typing import Any, Dict, List, Union
from functools import partial
print = partial(print, flush=True)

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Merge multiple lines of sequences to one line")
    p.add_argument('fasta')

    #p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    #np.random.seed(args.seed)
    fasta = dict()
    name, seq = None, list()
    with custom_open(args.fasta) as infile:
        for l in infile:
            if l.startswith('>'):
                if name is not None:
                    print("{}\n{}".format(name, ''.join(seq)))
                    name = None
                    seq = list()
                name = l.strip()
            else:
                seq.append(l.strip())

    print("{}\n{}".format(name, ''.join(seq)))
