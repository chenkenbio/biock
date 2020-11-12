#!/usr/bin/env python3

import os, sys, argparse
from gprofiler import GProfiler
import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('gene_list')
    p.add_argument('-s', '--species', choices=('hsapiens'), default='hsapiens')
    p.add_argument('-o', "--output", required=True)
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    gene_list = list(np.loadtxt(args.gene_list, dtype=str, comments='#'))
    #print(gene_list)
    gp = GProfiler(return_dataframe=True)
    out = gp.profile(organism=args.species, query=gene_list)
    out.to_csv(args.output, sep='\t', index=None)
