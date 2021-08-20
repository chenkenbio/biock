#!/usr/bin/env python3

import argparse, os, sys, time
# import logging, warnings, json, gzip, pickle
# warning = logging.warning


#from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from scipy.stats import pearsonr, spearmanr
from typing import Any, Dict, List, Union
from functools import partial
print = partial(print, flush=True)

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-a', required=True, help="if -b is not specified, assuming the first two rows in -a are the arrays")
    p.add_argument('-b', required=False)
    p.add_argument('-s', required=True, help="save name")
    p.add_argument('--xy-lims', nargs=4, default=(-1, 1, -1, 1))
    p.add_argument('-x', "--x-label", default="X")
    p.add_argument('-y', "--y-label", default="Y")
    p.add_argument('-f', "--fig-size", nargs=2, type=int, default=(5, 5))
    p.add_argument('--switch', action='store_true')

    #p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    #np.random.seed(args.seed)
    
    a, b = list(), list()
    if args.b is None:
        with open(args.a) as infile:
            for l in infile:
                if l.startswith('#'):
                    continue
                a_, b_ = l.split()[0:2]
                a.append(float(a_))
                b.append(float(b_))
    else:
        with open(args.a) as infile:
            for l in infile:
                if l.startswith('#'):
                    continue
                a_ = l.split()[0]
                a.append(float(a_))
        with open(args.b) as infile:
            for l in infile:
                if l.startswith('#'):
                    continue
                b_ = l.split()[0]
                b.append(float(b_))
    
    if args.switch:
        a, b = b, a
    
    plt.figure(figsize=args.fig_size)
    ax = plt.subplot()
    ax.scatter(a, b, s=1)
    ax.plot([-1, 1], [-1, 1], 'k:')
    ax.plot([-1, 1], [0, 0], 'k:')
    ax.plot([0, 0], [-1, 1], 'k:')
    ax.set_aspect('equal')
    ax.set_xlabel(args.x_label)
    ax.set_ylabel(args.y_label)
    ax.set_xlim(args.xy_lims[0:2])
    ax.set_ylim(args.xy_lims[2:4])
    plt.tight_layout()
    plt.savefig(args.s)
