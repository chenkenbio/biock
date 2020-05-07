#!/usr/bin/env python3

import argparse, os, sys, warnings, time, json, gzip
import numpy as np
import subprocess
from subprocess import Popen, PIPE

def make_directory(in_dir):
    if os.path.isfile(in_dir):
        warnings.warn("{} is a regular file".format(in_dir))
        return None
    outdir = in_dir.rstrip('/')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    return outdir


def prog_header(args=None, out=sys.stdout):
    print("\n# Started at {}".format(time.asctime()))
    print("## Command: {}".format(' '.join(sys.argv)), file=out)
    if args is not None:
        print("##: Args: {}".format(args))



def run_bash(cmd):
    p = Popen(['/bin/bash', '-c', cmd], stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    out, err = out.decode('utf8'), err.decode('utf8')
    rc = p.returncode
    return (rc, out, err)


## variables
hg19_chromsize = {"chr1": 249250621, "chr10": 135534747, "chr11": 135006516, 
        "chr12": 133851895, "chr13": 115169878, "chr14": 107349540, 
        "chr15": 102531392, "chr16": 90354753, "chr17": 81195210, 
        "chr18": 78077248, "chr19": 59128983, "chr2": 243199373, 
        "chr20": 63025520, "chr21": 48129895, "chr22": 51304566, 
        "chr3": 198022430, "chr4": 191154276, "chr5": 180915260, 
        "chr6": 171115067, "chr7": 159138663, "chr8": 146364022, 
        "chr9": 141213431, "chrM": 16569, 
        "chrX": 155270560, "chrY": 59373566}


if __name__ == "__main__":
    #args = get_args()
    pass



#import argparse, os, sys, warnings, time
#import numpy as np
#
#def get_args():
#    p = argparse.ArgumentParser()
#    return p.parse_args()
#
#if __name__ == "__main__":
#    args = get_args()
#
