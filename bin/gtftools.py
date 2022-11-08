#!/usr/bin/env python3

import argparse
import os
import sys
from biock import gtf_to_bed

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = p.add_subparsers(title="subparsers")
    fetch = subparsers.add_parser('fetch')
    fetch.add_argument("gtf")
    fetch.add_argument("--sep", '-s', default='|', help="delimiter character")
    fetch.add_argument("--attributes", '-a', nargs='+', required=True)
    # p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    args = get_args().parse_args()
    if args.gtf is not None:
        gtf_to_bed(args.gtf, attrs=args.attributes, sep=args.sep)
