#!/usr/bin/env python3

import os, sys, argparse
from gprofiler import GProfiler
import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('gene_list')
    p.add_argument('-s', '--species', choices=('hsapiens', 'mmuslus'), default='hsapiens')
    p.add_argument('--ordered', action='store_true')
    p.add_argument('-o', "--output", required=True)
    p.add_argument('-c', '--col-index', default=0,type=int) 
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    inputs = np.loadtxt(args.gene_list, dtype=str, comments='#') # .T[args.col_index]
    if len(inputs.shape) > 1:
        gene_list = list(inputs.T[args.col_index])
    else:
        gene_list = list(inputs)
    print("# {} genes: {} ...".format(len(gene_list), gene_list[0:10]))
    gp = GProfiler(return_dataframe=True)
    out = gp.profile(organism=args.species, sources=['GO'], query=gene_list, ordered=args.ordered)
    out.to_csv(args.output, sep='\t', index=None)
