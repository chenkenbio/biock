#!/usr/bin/env python3

import argparse, os, sys, time
#import warnings, json, gzip, pickle
#from collections import defaultdict, OrderedDict

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from scipy.stats import pearsonr, spearmanr
from typing import Any, Dict, List, Union
from functools import partial
print = partial(print, flush=True)

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', "--input", nargs='+', required=True)
    p.add_argument('-c', "--columns", required=True, nargs='+', type=int, help="columns (0-based)")
    p.add_argument('-d', default=None, help="Default is tab")
    p.add_argument('--detail', action='store_true')
    #p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    #np.random.seed(args.seed)
    
    if args.d is None:
        delimiter = '\t'
    else:
        delimiter = args.d
    s = list()
    for fn in args.input:
        with open(fn) as infile:
            for l in infile:
                fields = l.strip('\n').split(delimiter)
                for i in args.columns:
                    s.append(fields[i])
    if args.detail:
        s = sorted([(a, b) for a, b in zip(*np.unique(s, return_counts=True))], key=lambda l:l[1], reverse=True)
        for a, b in s:
            print("{}\t{}".format(a, b))
    else:
        print(len(set(s)))

