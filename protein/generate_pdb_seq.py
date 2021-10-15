#!/usr/bin/env python3

import argparse, os, sys, time, tqdm
import logging, warnings, json, gzip, pickle
# warning = logging.warning
from .pdb import PDB


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
    inputs = p.add_mutually_exclusive_group()
    inputs.add_argument("-list")
    inputs.add_argument("-pdb", nargs='+')
    #p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    #np.random.seed(args.seed)

    if args.list is not None:
        inputs = sorted(list(json.load(open(args.list))))
    else:
        inputs = list(args.pdb)

    for fn in tqdm.tqdm(inputs):
        pdb = os.path.basename(fn).split('.')[0][3:]
        assert len(pdb) == 4, pdb
        pdbobj = libpdb.PDB(fn)
        for chain, seq in pdbobj.chains.items():
            print(">{}{}|RESO={}\n{}".format(pdbobj.pdb_id, chain, pdbobj.resolution, ''.join(seq)))

