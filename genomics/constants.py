#!/usr/bin/env python3

import os, json
import numpy as np
from typing import Any, Dict, Iterable, List, Literal, Optional, Union
# from .single_cell import keep_common_cells, random_adata

HOME = os.environ["HOME"]


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
HG38_FASTA_H5 = os.path.join(HOME, "db/gencode/GRCh38/GRCh38.primary_assembly.genome.fa.h5")
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
NN_COMPLEMENT_INT = np.array([0, 4, 3, 2, 1])

