#!/usr/bin/env python3

from glob import glob
import argparse, os, sys, warnings, time, json, gzip, logging, warnings, pickle
from collections import OrderedDict
from argparse import Namespace
import shutil
import inspect
import stat
import filecmp
import random, string
import numpy as np
from io import TextIOWrapper
import subprocess
from subprocess import Popen, PIPE
from typing import Any, Dict, List, Text, TextIO, Union, Iterable, Tuple
import functools
import logging
logger = logging.getLogger(__name__)
# from biock.genomics import HG38_FASTA, HG19_FASTA

print = functools.partial(print, flush=True)
print_err = functools.partial(print, flush=True, file=sys.stderr)

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
# if os.path.exists(os.path.join(FILE_DIR, "variables.py")):

### misc

def scientific_notation(x, decimal: int=3):
    template = "{:." + str(decimal) + "e}"
    number, exp = template.format(x).split('e')
    exp = int(exp)
    return r"$%s\times 10^{%d}$" % (number, exp)

def count_dict_value(d: Dict[Any, int], sort=True, reverse=True, decimal=3) -> List[Tuple[Any, int, float]]:
    total = sum(d.values())
    ar = [(k, v, v/total) for k, v in d.items()]
    if sort:
        ar = sorted(ar, key=lambda x:x[1], reverse=reverse)
    ar = [(k, v, round(f, decimal)) for k, v, f in ar]
    return ar


def str2dict(s: str, delimiter: str, assign_char: str='=', strip_key: str=None, strip_value: str=None) -> Dict[str, Union[str, List[str]]]:
    d = dict()
    for kv in s.split(delimiter):
        k, v = kv.split(assign_char)
        if strip_key is not None:
            k = k.strip(strip_key)
        if strip_value is not None:
            v = v.strip(strip_value)
        if k in d:
            if not isinstance(d[k], list):
                d[k] = [d[k]]
            d[k].append(v)
        else:
            d[k] = v
    return d


def count_items(ar: List, sort_counts: bool=False, reverse: bool=True, fraction: bool=False):
    ar = np.asarray(ar)
    if sort_counts:
        results = sorted(zip(*np.unique(ar, return_counts=True)), key=lambda x:x[1], reverse=reverse)
    else:
        results = list(zip(*np.unique(ar, return_counts=True)))
    if fraction:
        total = len(ar)
        results = [list(x) + [round(x[1] / total, 3)] for x in results]
    return results

def pickle_dump(obj: Any, output, compression=None, force=False):
    if os.path.exists(output) and not force:
        raise IOError(r"File {obj} exists, use `force=True` to overwrite.")
    if compression is None:
        custom_open = functools.partial(open, mode='wb')
    elif compression == 'gzip':
        custom_open = functools.partial(gzip.open, mode='wb')
    else:
        raise KeyError("Unrecognized compression option: {}".format(compression))
    pickle.dump(obj, custom_open(output), protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(obj):
    raise NotImplementedError

def to_rank(x):
    raw_idx = [t[1] for t in sorted(zip(x, np.arange(len(x))), key=lambda ar:ar[0], reverse=False)]
    new_score = [t[1] for t in sorted(zip(raw_idx, np.arange(1, 1 + len(raw_idx))/len(raw_idx)), key=lambda ar:ar[0])]
    return np.asarray(new_score)

def string2dict(s, sep=';', assign='=') -> Dict[str, str]:
    d = dict()
    unknown_count = 0
    for kv in s.split(sep):
        try:
            k, v = kv.split(assign)
        except:
            k = "unknown-key_{}".format(unknown_count)
            unknown_count += 1
        d[k] = v
    return d


def md5_file(fn):
    import hashlib
    return hashlib.md5(open(fn, 'rb').read()).hexdigest()


def has_module(module):
    import imp
    exist = None
    try:
        imp.find_module(module)
        exist = True
    except ImportError:
        exist = False
    return exist


def hash_string(s):
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()

def merge_intervals(intervals: List[List[int]], col1: int, col2: int) -> List[List[int]]:
    intervals = [list(x) for x in intervals]
    intervals = sorted(intervals, key=lambda x:(x[col1], x[col2]))
    merged = list()
    merged.append(intervals[0])
    for x in intervals[1:]:
        l, r = x[col1], x[col2]
        if l <= merged[-1][col2]:
            merged[-1][col2] = r
        else:
            merged.append(x)
    return merged

def str2num(s: str) -> Union[int, float]:
    s = s.strip()
    try:
        n = int(s)
    except:
        n = float(s)
    return n


def auto_open(input: Union[str, TextIOWrapper], mode='rt') -> TextIOWrapper:
    if isinstance(input, str):
        if input == '-':
            return sys.stdin
        elif input.endswith(".gz") or input.endswith(".bgz"):
            return gzip.open(input, mode=mode)
        else:
            return open(input, mode=mode)
    elif isinstance(input, TextIOWrapper):
        return input
    else:
        raise IOError("Unknown input type {}".format(type(input)))
copen = auto_open # copen: custom open


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

def overlap_length2(x1, x2, y1, y2):
    """ [x1, x2), [y1, y2) """
    length = 0
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    if x2 <= y1:
        length = x2 - y1
    elif x1 >= y2:
        length = y2 - x1
    else:
        length = min(x2, y2) - max(x1, y1)
    return length



def distance(x1, x2, y1, y2, nonoverlap=False):
    """ interval distance """
    d = overlap_length(x1, x2, y1, y2)
    if nonoverlap and d < 0:
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

# def get_logger(log_level="INFO"):
#     logging.basicConfig(
#                 format='[%(asctime)s %(levelname)s] %(message)s',
#                     stream=sys.stdout
#             )
#     log = logging.getLogger(__name__)
#     log.setLevel(log_level)
#     return log


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

def remove_files(filenames: Union[str, Iterable[str]]) -> int:
    """
    Args:
        filenames: filename/regex/file list
    Return
        cnt : number of deleted files
    """
    if type(filenames) is str:
        if '*' in filenames:
            filenames = glob(filenames)
        else:
            filenames = [filenames]
    cnt = 0
    for fn in filenames:
        if os.path.exists(fn):
            os.remove(fn)
            cnt += 1
    return cnt

def run_bash(cmd) -> Tuple[int, str, str]:
    r"""
    Return
    -------
    rc : return code
    out : output
    err : error
    """
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




def load_chrom_size(fai) -> Dict[str, int]:
    chromsize = dict()
    with open(fai) as infile:
        for l in infile:
            chrom, size = l.strip().split('\t')[:2]
            chromsize[chrom] = int(size)
    return chromsize



def array_summary(x):
    x = np.array(x)
    r = {'mean': np.mean(x).round(3), 'min': np.round(min(x), 3)}
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        r[q] = np.quantile(x, q).round(3)
    r['max'] = np.round(max(x), 3)
    return r

def random_string(n):
    random.seed(time.time() % 3533)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))


def make_readonly(filename: str):
    mode = os.stat(filename).st_mode
    ro_mask = 0o777 ^ (stat.S_IWRITE | stat.S_IWGRP | stat.S_IWOTH)
    os.chmod(filename, mode & ro_mask)


def backup_file(src, dst, readonly: bool=False, **kwargs) -> str:
    r"""
    Parameters
    -----------
    src : source file to be backup
    dst : destination directory/filename
    safe : make dst readonly

    Return
    -------
    dst name or renamed dst name (when collision happens)
    """
    if "safe" in kwargs:
        readonly = kwargs["safe"]
        warnings.warn("`safe` should be replaced with `readonly`")
    if inspect.ismodule(src):
        src = src.__file__
    elif inspect.isclass(src):
        src = inspect.getfile(src)
        
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    bn = os.path.basename(dst)
    if '.' in bn:
        suffix = bn.split('.')[-1]
    else:
        suffix = ""
    if os.path.exists(dst):
        if not filecmp.cmp(src, dst):
            stamp = '.'.join([time.strftime("%Y_%m_%d-%H_%M_%S"), suffix])
            # dst = dst + stamp
            while os.path.exists(dst + '.' + stamp):
                stamp = '.'.join([time.strftime("%Y_%m_%d-%H_%M_%S"), suffix])
            dst = dst + '.' + stamp
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)
    if readonly:
        make_readonly(dst)
    return dst



def wait_memory(min_memory=64, max_try: int=100, cycle_time=5):
    cycle_time = cycle_time * 60
    import psutil
    free_mem = psutil.virtual_memory().available / (1024**3)
    cnt = 0
    while free_mem < min_memory and cnt < max_try:
        cnt += 1
        print("WARNING: inadequate memory! (required/available, GB): {:.1f}/{:.1f}, waitting {}s ...({}/{}, {})".format(min_memory, free_mem, cycle_time, cnt, max_try, time.asctime()), file=sys.stderr)
        time.sleep(cycle_time)
        free_mem = psutil.virtual_memory().available / (1024**3)
    if free_mem >= min_memory:
        enough = True
    else:
        enough = False
        raise MemoryError("Timeout, inadquate memory!".format(max_try))
    return enough


def check_exists(*argv):
    missing = list()
    for fn in argv:
        if fn is None:
            continue
        elif not os.path.exists(fn):
            missing.append(fn)
    if len(missing) > 0:
        raise FileExistsError("Missing file(s): {}".format(', '.join(missing)))


def check_args(args: Namespace, *keys):
    is_none = list()
    for k in keys:
        if args.__dict__[k] is None:
            is_none.append(k)
    if len(is_none) > 0:
        raise ValueError("{} is(are) None".format('/'.join(["args.{}".format(k) for k in  is_none])))




def get_run_info(argv: List[str], args: Namespace=None, **kwargs) -> str:
    s = list()
    s.append("")
    s.append("##time: {}".format(time.asctime()))
    s.append("##cwd: {}".format(os.getcwd()))
    s.append("##cmd: {}".format(' '.join(argv)))
    if args is not None:
        s.append("##args: {}".format(args))
    for k, v in kwargs.items():
        s.append("##{}: {}".format(k, v))
    return '\n'.join(s)



class LabelEncoder(object):
    def __init__(self, predefined_mapping: Dict[str, int]=dict()) -> None:
        self.mapping = predefined_mapping.copy()
        if len(self.mapping) == 0:
            self._next = 0
        else:
            self._next = max(self.mapping.values()) + 1
        self.reverse_mapping = {v:k for k, v in self.mapping.items()}
    
    def __call__(self, label) -> int:
        if label not in self.mapping:
            self.mapping[label] = self._next
            self.reverse_mapping[self._next] = label
            self._next += 1
        return self.mapping[label]

    def id2label(self, id) -> str:
        return self.reverse_mapping[id]
        

def np_onehot(ar, num_classes: int):
    pass


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



