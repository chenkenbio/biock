#!/usr/bin/env python3

import argparse, os, sys, warnings, time, json, gzip
import numpy as np

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('bed')
    p.add_argument('-l', '--length', type=int, default=3000)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()

    with open(args.bed) as infile:
        for l in infile:
            fields = l.strip().split()
            chrom, start, end = fields[0:3]
            mid = (int(start) + int(end)) // 2
            print("{}\t{}\t{}{}".format(
                chrom, mid - args.length // 2, mid +  args.length // 2, '\t{}'.format('\t'.join(fields[3:]) if len(fields) > 3 else "")
                ))


