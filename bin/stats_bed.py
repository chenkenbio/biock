#!/usr/bin/env python3
"""
Author: Ken Chen (chenkenbio@gmail.com)
Date: <<date>>
"""

import argparse, sys
import numpy as np
from biock import copen
from collections import defaultdict

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Count interval size")
    p.add_argument('bed')
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()

    starts, ends = defaultdict(list), defaultdict(list)

    if args.bed == '-':
        for l in sys.stdin:
            if l.startswith("#"):
                continue
            chrom, start, end = l.strip().split('\t')[0:3]
            starts[chrom].append(int(start))
            ends[chrom].append(int(end))
    else:
        with copen(args.bed) as infile:
            for l in infile:
                if l.startswith("#"):
                    continue
                chrom, start, end = l.strip().split('\t')[0:3]
                starts[chrom].append(int(start))
                ends[chrom].append(int(end))

    total = 0
    for chrom in starts:
        starts[chrom] = np.asarray(starts[chrom])
        ends[chrom] = np.asarray(ends[chrom])
        length = ends[chrom] - starts[chrom]
        print("{}(n/min/mean/max/sum)\t{:d}\t{}".format(chrom, len(length), '\t'.join(["{:d}".format(x) for x in (length.min(), round(length.mean()), length.max(), length.sum())])))
        total += length.sum()
    print("total\t{:d}".format(total))


