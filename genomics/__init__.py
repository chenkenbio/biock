#!/usr/bin/env python3

import os, json
from .toolbox import parse_gtf_record, chrom_remove_chr, ensembl_remove_version, chrom_add_chr, fix_peak_name, load_gene_info
import numpy as np
from typing import Any, Dict, Iterable, List, Literal, Optional, Union
from scipy.sparse import issparse, csr_matrix
# from .single_cell import keep_common_cells, random_adata
import warnings

HOME = os.environ["HOME"]

def _counts_per_size(mtx: np.ndarray, log: bool=False, target_reads: int=1e6) -> np.ndarray:
    """
    Args:
        mtx : cell by gene matrix
    Return:
        cpm/logcpm
    """
    size = np.asarray(mtx).sum(axis=1)
    cpm = ((target_reads / size) * mtx.T).T
    if log:
        cpm = np.log1p(cpm)
    return cpm

def counts_per_thousand(mtx, log: bool=False) -> np.ndarray:
    return _counts_per_size(mtx, log, target_reads=1000)
def counts_per_million(mtx, log: bool=False) -> np.ndarray:
    return _counts_per_size(mtx, log, target_reads=1E6)

def _reverse_counts_per_scale(mtx, libsize, target_reads) -> np.ndarray:
    raise NotImplementedError

def reverse_counts_per_million():
    raise NotImplementedError
    return _reverse_counts_per_scale()


__chrom_sizes__ = json.load(open(os.path.join(os.path.dirname(__file__), "chrom_sizes.json")))

HG19_CHROMSIZE = __chrom_sizes__["HG19_CHROMSIZE"]
HG38_CHROMSIZE = __chrom_sizes__["HG38_CHROMSIZE"]
MM10_CHROMSIZE = __chrom_sizes__["MM10_CHROMSIZE"]
del __chrom_sizes__

CHROM_SIZE_DICT = {
        'hg19': HG19_CHROMSIZE, 'GRCh37': HG19_CHROMSIZE, 
        "hg38": HG38_CHROMSIZE, "GRCh38": HG38_CHROMSIZE,
        "hg38-shuffle": HG38_CHROMSIZE, 
        "hg38-random": HG38_CHROMSIZE, 
        "mm10": MM10_CHROMSIZE
    }

HG19_FASTA = os.path.join(HOME, "db/gencode/GRCh37/GRCh37.primary_assembly.genome.fa")
HG38_FASTA = os.path.join(HOME, "db/gencode/GRCh38/GRCh38.primary_assembly.genome.fa")
MM9_FASTA = os.path.join(HOME, "db/UCSC/mm9/mm9.fa")
MM10_FASTA = os.path.join(HOME, "db/UCSC/mm10/mm10.fa")
MM39_FASTA = os.path.join(HOME, "db/gencode/GRCm39/GRCm39.primary_assembly.genome.fa")


HG19_SIZE_FN = HG19_FASTA + ".chromsize"
HG38_SIZE_FN = HG38_FASTA + ".chromsize"
MM39_SIZE_FN = MM39_FASTA + ".chromsize"

HG19_FASTA_INT8 = os.path.join(HOME, "db/gencode/GRCh37/GRCh37.primary_assembly.genome.int8.pkl")
HG38_FASTA_INT8 = os.path.join(HOME, "db/gencode/GRCh38/GRCh38.primary_assembly.genome.int8.pkl")
# MM10_FASTA_INT8 = os.path.join(HOME, "db/gencode/GRCm38/GRCm38.primary_assembly.genome.int8.pkl")
MM10_FASTA_INT8 = os.path.join(HOME, "db/UCSC/mm10/mm10.int8.pkl")
MM9_FASTA_INT8  = os.path.join(HOME, "db/UCSC/mm9/mm9.int8.pkl")


FASTA_DICT = {
        'hg19': HG19_FASTA, 'GRCh37': HG19_FASTA, 
        'hg38': HG38_FASTA, 'GRCh38': HG38_FASTA,
        "mm10": MM10_FASTA
    }

FASTA_INT8_DICT = {
        'hg19': HG19_FASTA_INT8, 'GRCh37': HG19_FASTA_INT8, 
        'hg38': HG38_FASTA_INT8, 'GRCh38': HG38_FASTA_INT8,
        "mm10": MM10_FASTA_INT8
    }


_CHROM2INT = {
    "chr1": 1,   "1": 1,   1: 1,
    "chr2": 2,   "2": 2,   2: 2,
    "chr3": 3,   "3": 3,   3: 3,
    "chr4": 4,   "4": 4,   4: 4,
    "chr5": 5,   "5": 5,   5: 5,
    "chr6": 6,   "6": 6,   6: 6,
    "chr7": 7,   "7": 7,   7: 7,
    "chr8": 8,   "8": 8,   8: 8,
    "chr9": 9,   "9": 9,   9: 9,
    "chr10": 10, "10": 10, 10: 10,
    "chr11": 11, "11": 11, 11: 11,
    "chr12": 12, "12": 12, 12: 12,
    "chr13": 13, "13": 13, 13: 13,
    "chr14": 14, "14": 14, 14: 14,
    "chr15": 15, "15": 15, 15: 15,
    "chr16": 16, "16": 16, 16: 16,
    "chr17": 17, "17": 17, 17: 17,
    "chr18": 18, "18": 18, 18: 18,
    "chr19": 19, "19": 19, 19: 19,
    "chr20": 20, "20": 20, 20: 20,
    "chr21": 21, "21": 21, 21: 21,
    "chr22": 22, "22": 22, 22: 22,
    "chrX": 23,  "X": 23,
    "chrY": 24,  "Y": 24,
    "chrM": 25,  "M": 25
}
class Chrom2Int(object):
    def __init__(self) -> None:
        self.mapping = _CHROM2INT.copy()
        self.reverse_mapping = {v:k for k, v in self.mapping.items()}
        self._next = max(self.mapping.values()) + 1

    def __call__(self, chrom) -> int:
        if chrom not in self.mapping:
            self.mapping[chrom] = self._next
            self.reverse_mapping[self._next] = chrom
            self._next += 1
        return self.mapping[chrom]
    
    def int2chrom(self, idx):
        return self.reverse_mapping[idx]
        

def chrom2int(chrom):
    if chrom in _CHROM2INT:
        chrom = _CHROM2INT[chrom]
    return chrom

NN_COMPLEMENT = {
    'A': 'T', 'a': 't',
    'C': 'G', 'c': 'g',
    'G': 'C', 'g': 'c',
    'T': 'A', 't': 'a',
    'R': 'Y', 'r': 'y',
    'Y': 'R', 'y': 'r',
    'S': 'S', 's': 's',
    'W': 'W', 'w': 'w',
    'K': 'M', 'k': 'm',
    'M': 'K', 'm': 'k',
    'B': 'V', 'b': 'v',
    'D': 'H', 'd': 'h',
    'H': 'D', 'h': 'd',
    'V': 'B', 'v': 'b',
    'N': 'N', 'n': 'n'
}

def get_reverse_strand(seq, join: bool=True):
    if join:
        seq = ''.join([NN_COMPLEMENT.get(n, n) for n in seq[::-1]])
    else:
        seq = [NN_COMPLEMENT.get(n, n) for n in seq[::-1]]
    return seq


# from .transcripts import load_transcripts, Transcript, AnnotatedTranscript


def load_chrom_size(fai) -> Dict[str, int]:
    chrom_size = dict()
    with open(fai) as infile:
        for l in infile:
            chrom, size = l.split()[:2]
            chrom_size[chrom] = int(size)
    # if chrom.startswith("chr"):
    #     chrom_size[chrom[3:]] = int(size)
    return chrom_size

