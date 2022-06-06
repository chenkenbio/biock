#!/usr/bin/env python3

import argparse
from tqdm import tqdm
import pickle
import os
import sys
import numpy as np
import pandas as pd
from biock import load_fasta
import logging
logger = logging.getLogger(__name__)


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-g', "--genome", nargs='+', required=True)
    # p.add_argument("--gencode", action="store_true")
    p.add_argument("--output", '-o', required=True)
    return p


if __name__ == "__main__":
    args = get_args().parse_args()

    fasta = dict()
    for fn in tqdm(args.genome):
        d = load_fasta(fn)
        for k, v in d.items():
            k = k.split()[0]
            logger.info("{}: {}".format(k, len(v)))
            assert k not in fasta, "{}".format((k, fasta.keys()))
            fasta[k] = v

    pickle.dump(fasta, open(args.output, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
