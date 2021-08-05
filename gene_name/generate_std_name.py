#!/usr/bin/env python3

import argparse, os, sys, time
import logging, json, gzip, pickle

from collections import defaultdict, OrderedDict

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from scipy.stats import pearsonr, spearmanr
from typing import Any, Dict, List, Union
from functools import partial
print = partial(print, flush=True)

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('source', help="e.g.: /home/chenken/db/NCBI_GENE_INFO/Homo_sapiens.gene_info.20210615.gz")
    p.add_argument('-t', "--type", choices=("symbol", "entrez"), required=True)

    #p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    #np.random.seed(args.seed)

    golden_standard = dict()
    gene_std_name = defaultdict(set)
    with gzip.open(args.source, 'rt') as infile:
        for l in infile:
            if l.startswith('#'):
                continue
            gene_id, std_name, _, synonyms, dbrefs = l.strip('\n').split('\t')[1:6]
            if args.type == "entrez":
                std_name, gene_id = gene_id, std_name
            golden_standard[std_name] = gene_id
            golden_standard[gene_id] = gene_id
            gene_std_name[std_name].add(std_name)
            gene_std_name[gene_id].add(std_name)

            if synonyms != '-':
                for name in synonyms.split('|'):
                    gene_std_name[name].add(std_name)
            if synonyms != '-':
                for name in dbrefs.split('|'):
                    if name.startswith("Ensemb") or name.startswith("HGNC"):
                        name = ':'.join(name.split(':')[1:])
                    gene_std_name[name].add(std_name)

    for name, std_names in gene_std_name.items():
        if len(std_names) > 1:
            if name in golden_standard:
                gene_std_name[name] = golden_standard[name]
                logging.warning("ambiguous name {}: {} -> {}".format(name, std_names, golden_standard[name]))
            else:
                gene_std_name[name] = ';'.join(sorted(list(std_names)))
                logging.warning("ambiguous name {}: {}, kept".format(name, std_names))
        else:
            gene_std_name[name] = ';'.join(sorted(list(std_names)))
    json.dump(gene_std_name, fp=sys.stdout, indent=4)

