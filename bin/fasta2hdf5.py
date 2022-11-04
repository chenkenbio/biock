#!/usr/bin/env python3

import argparse
from tqdm import tqdm
import os
import numpy as np
import h5py
from biock.variables import NUCLEOTIDE4
from biock import copen
import logging
logger = logging.getLogger(__name__)


def fasta2dict(fasta, token_table):
    sequences = dict()
    name = ""
    with copen(fasta) as infile:
        for l in tqdm(infile, desc=name):
            if l.startswith(">"):
                name = l.strip().split()[0].replace(">", '')
                sequences[name] = list()
            else:
                seq = l.strip()
                seq = np.asarray([token_table[n] for n in seq], dtype=np.int8)
                sequences[name].append(seq)
    for name, seqs in sequences.items():
        sequences[name] = np.concatenate(seqs)
        print(name, sequences[name].shape)
    return sequences


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("fasta", help="input")
    p.add_argument("-o", "--output", help="input")
    p.add_argument("-t", "--type", choices=("nucleotide", "protein"), default="nucleotide", required=False)
    # p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    args = get_args().parse_args()

    if os.path.exists(args.output):
        raise FileExistsError("file {} exists".format(args.output))
    if args.type == "protein":
        raise NotImplementedError("do not support protein")
    elif args.type == "nucleotide":
        table = NUCLEOTIDE4
    
    sequences = fasta2dict(args.fasta, table)

    fp = h5py.File(args.output, 'w')
    for name, seq in sequences.items():
        fp.create_dataset(name, data=np.asarray(seq, dtype=np.int8))

