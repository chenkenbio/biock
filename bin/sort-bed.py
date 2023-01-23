#!/usr/bin/env python3
"""
Author: Ken Chen (chenkenbio@gmail.com)
Date: <<date>>
"""

import argparse, os, sys, gzip
from biock import copen

def get_args():
    p = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
            description="Sorting bed files in numerical order"
        )
    p.add_argument('-i', "--input", help="Input, bed file or from pipe (-)", metavar="bed", required=True)
    return p

CHROM2INT = {
        "chr1": 1, "chr2": 2, "chr3": 3, "chr4": 4, "chr5": 5, "chr6": 6, 
        "chr7": 7, "chr8": 8, "chr9": 9, "chr10": 10, "chr11": 11, "chr12": 12, 
        "chr13": 13, "chr14": 14, "chr15": 15, "chr16": 16, "chr17": 17, 
        "chr18": 18, "chr19": 19, "chr20": 20, "chr21": 21, "chr22": 22, 
        "chrX": 23, "chrY": 24, "chrM": 25, "chrMT": 25,
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
        "10": 10, "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, 
        "17": 17, "18": 18, "19": 19, "20": 20, "21": 21, "22": 22, "X": 23, 
        "Y": 24, "M": 25, "MT": 25
    }

if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()

    if args.input is None or args.input == '-':
        inputs = sys.stdin
    else:
        inputs = copen(args.input)
    
    records = list()
    regular = list()
    irregular = list()

    for nr, l in enumerate(inputs):
        fields = l.strip().split('\t')
        chrom, start, end = fields[0:3]
        records.append(l.strip())
        if chrom in CHROM2INT:
            chrom = CHROM2INT[chrom]
            regular.append((chrom, int(start), int(end), nr))
        else:
            irregular.append((chrom, int(start), int(end), nr))

    if args.input is not None:
        inputs.close()

    regular = sorted(regular, key=lambda x:(x[0], x[1], x[2]))
    irregular = sorted(irregular, key=lambda x:(x[0], x[1], x[2]))

    for _, _, _, nr in regular:
        print(records[nr])
    for _, _, _, nr in irregular:
        print(records[nr])

