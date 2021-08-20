#!/usr/bin/env python3

import argparse, os, sys, time
import logging, warnings, json, gzip, pickle
# warning = logging.warning


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
    p.add_argument('-a', required=True)
    p.add_argument('-b', required=True)

    #p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    #np.random.seed(args.seed)

    try:
        a = set(json.load(open(args.a)))
    except:
        a = set()
        with open(args.a) as infile:
            for l in infile:
                a.add(l.strip())
    print("## a: {} ...".format(' '.join(list(a)[0:5])))

    try:
        b = set(json.load(open(args.b)))
    except:
        b = set()
        with open(args.b) as infile:
            for l in infile:
                b.add(l.strip())
    print("## b: {} ...".format(' '.join(list(b)[0:5])))


    print("#size_a\tsize_b\tintersection\tunion")
    print("{}\t{}\t{}\t{}".format(len(a), len(b), len(a.intersection(b)), len(a.union(b))))
