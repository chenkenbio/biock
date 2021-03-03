#!/usr/bin/env python3

import argparse, os, sys, time
#import warnings, json, gzip

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from scipy.stats import pearsonr, spearmanr

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('file')
    p.add_argument('-c', "--column", type=int, default=0)
    p.add_argument('--bed', required=True, nargs='+')
    p.add_argument('-a', "--append", action='store_true')
    p.add_argument('-r', "--replace", action='store_true')

    #p.add_argument('--seed', type=int, default=2020)
    return p.parse_args()

def clean_id(gene_id):
    if "_PAR_Y" in gene_id:
        gene_id = gene_id.split('.')[0] + "_PAR_Y"
    else:
        gene_id = gene_id.split('.')[0]
    return gene_id

def load_id_from_bed(beds):
    id2name = dict()
    for bed in beds:
        with open(bed) as infile:
            for l in infile:
                gene_id, gene_name, gene_type, tx_id = l.split('\t')[3].split('|')[0:4]
                if gene_id not in id2name:
                    id2name[gene_id] = gene_name
                    id2name[clean_id(gene_id)] = gene_name
                elif gene_name not in id2name[gene_id].split('|'):
                        id2name[gene_id] = "{}|{}".format(id2name[gene_id], gene_name)
                        id2name[clean_id(gene_id)] = "{}|{}".format(id2name[gene_id], gene_name)
    return id2name


if __name__ == "__main__":
    args = get_args()
    #np.random.seed(args.seed)
    id2name = load_id_from_bed(args.bed)
    with open(args.file) as infile:
        for l in infile:
            fields = l.strip().split('\t')
            gid = fields[args.column]
            if gid in id2name:
                gname = id2name[gid]
            else:
                gname = gid
            if args.replace:
                fields[args.column] = gname
                print('\t'.join(fields))
            elif args.append:
                fields.append(gname)
                print('\t'.join(fields))
            else:
                print(gname)

