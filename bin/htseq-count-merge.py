#!/usr/bin/env python3

import argparse, os, sys, time
import warnings, json, gzip

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from scipy.stats import pearsonr, spearmanr

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('counts', nargs='+')
    p.add_argument('--suffix', default=None)

    #p.add_argument('--seed', type=int, default=2020)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    #np.random.seed(args.seed)
    sample_list = list()
    gene_list = list()
    gene_expr = dict()
    for i, fn in enumerate(args.counts):
        sample = os.path.basename(fn)
        if args.suffix:
            sample = sample.replace(args.suffix, '')
        sample_list.append(sample)
        with gzip.open(fn, 'rt') as infile:
            for row, l in enumerate(infile):
                if l.startswith('_'):
                    continue
                gene_id, expr = l.strip().split()
                if i == 0:
                    gene_list.append(gene_id)
                    gene_expr[gene_id] = dict()
                gene_expr[gene_id][sample] = expr
    print("gene_id", end='')
    for sample in sample_list:
        print("\t{}".format(sample), end='')
    for gene_id in gene_list:
        print("\n{}".format(gene_id), end='')
        for sample in sample_list:
            print("\t{}".format(gene_expr[gene_id][sample]), end='')

