#!/usr/bin/env python3

import argparse, os, sys, warnings, time, json
#from biock.biock import run_bash
#import biock.biock as biock 

def parse_gtf_attr(attrs, category='gencode'):
    kv = dict()
    if category == 'genecode':
        for pair in attrs.rstrip(';').split('; '):
            k, v = pair.split(' ')
            kv[k] = v.strip('"')
    return kv


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('gtf')
    #p.add_argument('--choice', choices=('TSS'), default='TSS')
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    with open(args.gtf) as infile:
        for l in infile:
            if l.startswith('#'):
                continue
            chrom, _, feature_type, start, end, _, strand, _, attrs = l.strip().split('\t')
            if feature_type != 'transcript':
                continue
            attrs = parse_gtf_attr(attrs)
            gene_id = attrs['gene_id']
            tx_id = attrs['transcript_id']
            if strand == '-':
                tss = int(end) - 1
            else:
                tss = int(start)
            print("{}\t{}\t{}\t{}|{}\t.\t{}".format(chrom, tss, tss + 1, gene_id, tx_id, strand))

