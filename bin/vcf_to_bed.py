#!/usr/bin/env python3
"""
Author: Ken Chen
Email: chenkenbio@gmail.com
Date: 2022-12-11
"""

import argparse
import os
import sys
import numpy as np
from biock.genomics._vcf import VCFData

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("vcf")
    # p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    args = get_args().parse_args()

    vcf_data = VCFData(args.vcf)

    for r in vcf_data.to_bed():
        print("\t".join([str(x) for x in r]))

