#!/usr/bin/env python3

import argparse, os, sys, warnings, time, json, gzip, logging, warnings, pickle
import random, string
import numpy as np
from io import TextIOWrapper
import subprocess
from subprocess import Popen, PIPE
from typing import Any, Dict, List, Text, TextIO, Union
import functools

print = functools.partial(print, flush=True)
print_err = functools.partial(print, flush=True, file=sys.stderr)

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
# if os.path.exists(os.path.join(FILE_DIR, "variables.py")):

### misc
def md5_file(fn):
    import hashlib
    return hashlib.md5(open(fn, 'rb').read()).hexdigest()


def hash_string(s):
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()


def str2num(s: str) -> Union[int, float]:
    s = s.strip()
    try:
        n = int(s)
    except:
        n = float(s)
    return n


def copen(fn: str, mode='rt') -> TextIOWrapper:
    if fn.endswith(".gz"):
        return gzip.open(fn, mode=mode)
    else:
        return open(fn, mode=mode)
custom_open = copen


def strip_ENS_version(ensembl_id: str) -> str:
    suffix = ""
    if ensembl_id.endswith("_PAR_Y"):
        suffix = "_PAR_Y"
    return "{}{}".format(ensembl_id.split('.')[0], suffix)
remove_ENS_version = strip_ENS_version


def jaccard_sim(a, b):
    a, b = set(a), set(b)
    return len(a.intersection(b)) / len(a.union(b))


def overlap_length(x1, x2, y1, y2):
    """ [x1, x2), [y1, y2) """
    length = 0
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    if x2 <= y1:
        length = x2 - y1
    elif x1 <= y2:
        length = min(x2, y2) - max(x1, y1)
    else:
        length = y2 - x1
    return length


def distance(x1, x2, y1, y2):
    """ interval distance """
    d = overlap_length(x1, x2, y1, y2)
    if d < 0:
        warnings.warn("[{}, {}) overlaps with [{}, {})".format(x1, x2, y1, y2))
    return max(-d, 0)

def label_count(labels):
    """ labels should be list,np.array """
    categories, counts = np.unique(labels, return_counts=True)
    ratio = (counts / counts.sum()).round(3)
    return list(zip(categories, counts, ratio))


def split_chrom_start_end(chrom_start_end):
    """
    deal with chrom:start-end format
    """
    chrom, start_end = chrom_start_end.split(':')
    start, end = start_end.split('-')
    return chrom, int(start), int(end)


def split_train_valid_test(groups, train_keys, valid_keys, test_keys=None):
    """
    groups: length N, the number of samples
    train
    """
    assert isinstance(train_keys, list)
    assert isinstance(valid_keys, list)
    assert test_keys is None or isinstance(test_keys, list)
    index = np.arange(len(groups))
    train_idx = index[np.isin(groups, train_keys)]
    valid_idx = index[np.isin(groups, valid_keys)]
    if test_keys is not None:
        test_idx = index[np.isin(groups, test_keys)]
        return train_idx, valid_idx, test_idx
    else:
        return train_idx, valid_idx


def pandas_df2dict(fn, delimiter='\t', **kwargs):
    if type(fn) is str:
        import pandas as pd
        kwargs["delimiter"] = delimiter
        df = pd.read_csv(fn, **kwargs)
    else:
        df = fn
    d = dict()
    for k in df.columns:
        d[k] = np.array(df[k])
    return d



### logs
def print_run_info(args=None, out=sys.stdout):
    print("\n# PROG: '{}' started at {}".format(os.path.basename(sys.argv[0]), time.asctime()), file=out)
    print("## PWD: %s" % os.getcwd(), file=out)
    print("## CMD: %s" % ' '.join(sys.argv), file=out)
    if args is not None:
        print("## ARG: {}".format(args), file=out)

def prog_header(args=None, out=sys.stdout):
    print("\n# Started at {}".format(time.asctime()))
    print("## Command: {}".format(' '.join(sys.argv)), file=out)
    if args is not None:
        print("##: Args: {}".format(args))

def get_logger(log_level="INFO"):
    logging.basicConfig(
                format='[%(asctime)s %(levelname)s] %(message)s',
                    stream=sys.stdout
            )
    log = logging.getLogger(__name__)
    log.setLevel(log_level)
    return log


### file & directory
def dirname(fn):
    folder = os.path.basename(fn)
    if folder == '':
        folder = "./"
    return folder

def make_directory(in_dir):
    if os.path.isfile(in_dir):
        warnings.warn("{} is a regular file".format(in_dir))
        return None
    outdir = in_dir.rstrip('/')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    return outdir

def run_bash(cmd):
    p = Popen(['/bin/bash', '-c', cmd], stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    out, err = out.decode('utf8'), err.decode('utf8')
    rc = p.returncode
    return (rc, out, err)


class BasicBED(object):
    def __init__(self, input_file, bin_size=50000):
        self.input_file = input_file
        self.chroms = dict()
        self.bin_size = bin_size
        #self.parse_input()

    def intersect(self, chrom, start, end, gap=0):
        start, end = int(start) - gap, int(end) + gap
        if start >= end:
            warnings.warn("starat >= end: start={}, end={}".format(start, end))
        res = set()
        if chrom in self.chroms:
            for idx in range(start // self.bin_size, (end - 1) // self.bin_size + 1):
                if idx not in self.chroms[chrom]:
                    continue
                try:
                    for i_start, i_end, attr in self.chroms[chrom][idx]:
                        if i_start >= end or i_end <= start:
                            continue
                        res.add((i_start, i_end, attr))
                except:
                    print(self.chroms[chrom][idx])
                    exit(1)
        res = sorted(list(res), key=lambda l:(l[0], l[1]))
        return res

    def sort(self, merge=False):
        for chrom in self.chroms:
            for idx in self.chroms[chrom]:
                self.chroms[chrom][idx] = \
                        sorted(self.chroms[chrom][idx], key=lambda l:(l[0], l[1]))

    def add_record(self, chrom, start, end, attrs=None, cut=False):
        start, end = int(start), int(end)
        if chrom not in self.chroms:
            self.chroms[chrom] = dict()
        for bin_idx in range(start // self.bin_size, (end - 1) // self.bin_size + 1):
            if bin_idx not in self.chroms[chrom]:
                self.chroms[chrom][bin_idx] = list()
            if cut:
                raise NotImplementedError
            else:
                self.chroms[chrom][bin_idx].append((start, end, attrs))


    def __str__(self):
        return "BasicBED(filename:{})".format(os.path.relpath(self.input_file))

    def parse_input(self):
        raise NotImplementedError
        ## demo
        # with open(self.input_file) as infile:
        #     for l in infile:
        #         if l.startswith("#"):
        #             continue
        #         fields = l.strip('\n').split('\t')
        #         chrom, start, end = fields[0:3]
        #         self.add_record(chrom, start, end, attrs=fields[3:])
        # record format: (left, right, (XXX))
        # XXX: self defined attributes of interval [left, right)


class BasicFasta(object):
    def __init__(self, fasta):
        self.fasta = os.path.realpath(fasta)
        self.cache = self.fasta + ".pkl.gz"
        self.id2seq = dict()
        self.__load_fasta()
        self.num_seqs = len(self.id2seq)

    def extract_seq_id(self, header):
        raise NotImplementedError

    def __load_fasta(self):
        seq = str()
        if os.path.exists(self.cache):
            self.id2seq = pickle.load(gzip.open(self.cache, 'rb'))
        else:
            open_file = gzip.open if self.fasta.endswith('gz') else open
            with open_file(self.fasta, 'rt') as infile:
                for l in infile:
                    if l.startswith('>'):
                        if len(seq) > 0:
                            self.id2seq[seq_id] = seq
                            seq = str()
                        seq_id = self.extract_seq_id(l)
                    else:
                        seq += l.strip()
                self.id2seq[seq_id] = seq
            pickle.dump(self.id2seq, gzip.open(self.cache, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def load_fasta(fn: str) -> Dict[str, str]:
    fasta = dict()
    name, seq = None, list()
    with custom_open(fn) as infile:
        for l in infile:
            if l.startswith('>'):
                if name is not None:
                    # print("{}\n{}".format(name, ''.join(seq)))
                    fasta[name] = ''.join(seq)
                name = l.strip().lstrip('>')
                seq = list()
            else:
                seq.append(l.strip())
    fasta[name] = ''.join(seq)
    return fasta


def array_summary(x):
    x = np.array(x)
    r = {'mean': np.mean(x).round(3), 'min': np.round(min(x), 3)}
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        r[q] = np.quantile(x, q).round(3)
    r['max'] = np.round(max(x), 3)
    return r

def random_string(n):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))


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



