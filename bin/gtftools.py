#!/usr/bin/env python3

import argparse
import os
import sys
from biock import gtf_to_bed

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Fetch information from GTF annotation")
    subparsers = p.add_subparsers(title="command", dest="command")
    fetch = subparsers.add_parser('fetch', description="Convert gene/transcription/exon/... annotation to BED format")
    fetch.add_argument("-i", "--input", required=True, metavar="gtf", help="GTF file")
    fetch.add_argument("-f", "--feature-type", default="all")
    fetch.add_argument("--sep", '-s', default='|', help="delimiter character")
    fetch.add_argument("--attributes", '-a', nargs='+', required=True)
    # p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    if args.command is None:
        p.parse_args(['--help'])

    if args.input is not None:
        gtf_to_bed(args.gtf, args.feature_type, attrs=args.attributes, sep=args.sep, zero_start=False)
