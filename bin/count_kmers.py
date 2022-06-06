#!/usr/bin/env python3

import argparse
from tqdm import tqdm
import json
import os
import sys
import numpy as np
from collections import defaultdict
import pandas as pd
from biock import load_fasta


def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Count k-mers in fasta files"
    )
    p.add_argument('-fi', "--fasta", nargs='+', required=True, help="input fasta file(s)")
    p.add_argument("-k", default=1, type=int, help="size of k-mer (>=1)")
    p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    args = get_args().parse_args()

    counts = defaultdict(int)

    for fn in args.fasta:
        genomes = load_fasta(fn)
        for chrom, seq in genomes.items():
            length = len(seq)
            for i in tqdm(range(0, length - args.k), total=length - args.k, desc="{}:{}".format(os.path.basename(fn), chrom)):
                kmer = seq[i:i+args.k]
                counts[kmer] += 1
    
    json.dump(counts, fp=sys.stdout, indent=4)

