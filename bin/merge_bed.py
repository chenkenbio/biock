#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
import pandas as pd
import pybedtools
from pybedtools import BedTool, Interval
from biock import random_string, copen
import logging
logger = logging.getLogger(__name__)


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('bed', nargs='+')
    p.add_argument("--ignore-strand", action='store_true')
    p.add_argument("-d", type=int, default=0)
    p.add_argument("-o", required=True)
    # p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    args = get_args().parse_args()

    logger.setLevel("INFO")

    logger.info("## --ignore-strand mode: {}".format("OFF" if not args.ignore_strand else "ON"))

    forward = list()
    reverse = list()
    for fn in args.bed:
        if fn == '-':
            fn = sys.stdin
        with copen(fn) as infile:
            for nr, l in enumerate(infile):
                fields = l.strip().split('\t')
                if not args.ignore_strand:
                    chrom, start, end, _, _, strand = l.strip().split('\t')[:6]
                else:
                    strand = '+'
                    chrom, start, end = l.strip().split('\t')[:3]
                if strand == '+':
                    forward.append(Interval(chrom, int(start), int(end)))
                else:
                    reverse.append(Interval(chrom, int(start), int(end)))
    
    tmp_forward = "/tmp/pybedtools.{}.bed".format(random_string(8))
    BedTool(forward).sort().moveto(tmp_forward)
    BedTool(tmp_forward).merge(d=args.d).moveto(tmp_forward)
    if len(reverse) > 0:
        tmp_reverse = "/tmp/pybedtools.{}.bed".format(random_string(8))
        BedTool(reverse).sort().moveto(tmp_reverse)
        BedTool(tmp_reverse).merge(d=args.d).moveto(tmp_reverse)
    else:
        tmp_reverse = None
    
    bed = list()
    for strand, fn in [('+', tmp_forward), ('-', tmp_reverse)]:
        if fn is None:
            continue
        with open(fn) as infile:
            for l in infile:
                chrom, start, end = l.strip().split('\t')[:3]
                bed.append(Interval(chrom, int(start), int(end), strand=strand))
    
    BedTool(bed).sort().moveto(args.o)
    os.remove(tmp_forward)
    if tmp_reverse is not None:
        os.remove(tmp_reverse)



                
