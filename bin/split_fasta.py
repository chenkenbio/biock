#!/usr/bin/env python3

import argparse, os, sys, time
#import warnings, json, gzip

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from scipy.stats import pearsonr, spearmanr

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('fasta')
    p.add_argument('--outdir', required=True)

    #p.add_argument('--seed', type=int, default=2020)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    #np.random.seed(args.seed)
    with open(args.fasta) as infile:
        f = None
        for l in infile:
            if l.startswith('>'):
                header = l.lstrip('>').split(' ')[0].split('\t')[0]
                if f:
                    f.close()
                f = open(os.path.join(args.outdir, header + '.fa'), 'w')
                f.write(l)
            else:
                f.write(l)
        f.close()