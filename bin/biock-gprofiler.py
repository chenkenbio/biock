#!/usr/bin/env python3

import os, sys, argparse
from gprofiler import GProfiler
import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('gene_list')
    p.add_argument('-s', '--species', choices=('hsapiens', 'mmuslus'), default='hsapiens')
    p.add_argument('-o', "--output", required=True)
    p.add_argument('-c', '--col-index', default=0,type=int) 
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    gene_list = list(np.loadtxt(args.gene_list, dtype=str, comments='#').T[args.col_index])
    print("- Input genes: {} genes".format(len(gene_list)))
    #print(gene_list)
    gp = GProfiler(return_dataframe=True)
    out = gp.profile(organism=args.species, query=gene_list)
    out.to_csv(args.output, sep='\t', index=None)
