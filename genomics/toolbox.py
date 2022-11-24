#!/usr/bin/env python3
"""
Author: Ken Chen (chenkenbio@gmail.com)
Date: <<date>>
"""
import os
from tqdm import tqdm
import warnings
import sys
from collections import defaultdict, namedtuple, OrderedDict
from typing import Any, Union, List, Dict, Tuple, Literal
import numpy as np
import json
import logging
import pickle
import h5py
from .constants import NN_COMPLEMENT, NN_COMPLEMENT_INT, _CHROM2INT
from ..utils import copen
from ..variables import HUMAN_CHROMS_ALL
try:
    import pyfaidx
except ImportError as err:
    warnings.warn("Missing module: pyfaidx\n{}".format(err))

logger = logging.getLogger(__name__)

GTFRecord = namedtuple("GTFRecord", field_names=[
                       "chrom", "start", "end", "strand", "feature_type", "attrs"])


def parse_gtf_record(gtf_line: str, to_dict=False) -> GTFRecord:
    r"""
    Return:
    -------
    chrom : str
    feature_type: str
    start : int
    end : int
    strand : str
    attrs : dict
    """
    try:
        chrom, _, feature_type, start, end, _, strand, _, fields = gtf_line.strip(
        ).rstrip(';').split('\t')
    except:
        raise ValueError("failed to parse: {}".format(gtf_line.strip()))
    attrs = dict()
    for kv in fields.split('; '):
        v = kv.split(' ')
        k = v[0]
        v = ' '.join(v[1:])
        v = v.strip('"')
        if k in attrs:
            if isinstance(attrs[k], str):
                attrs[k] = [attrs[k]]
            attrs[k].append(v)
        else:
            attrs[k] = v
    if to_dict:
        return {
            "chrom":  chrom,
            "feature_type": feature_type,
            "start": int(start),
            "end": int(end),
            "strand": strand,
            "attrs": attrs
        }
    else:
        return GTFRecord(chrom=chrom, start=int(start), end=int(end), strand=strand, feature_type=feature_type, attrs=attrs)


def chrom_add_chr(chrom: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(chrom, str):
        chrom = "chr{}".format(chrom) if not chrom.startswith("chr") else chrom
    else:
        chrom = ['chr{}'.format(c) if not c.startswith(
            "chr") else c for c in chrom]
    return chrom


def chrom_remove_chr(chrom: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(chrom, str):
        chrom = chrom.replace("chr", '')
    else:
        chrom = [c.replace("chr", '') for c in chrom]
    return chrom


def chrom_add_chr(chrom: str) -> str:
    if isinstance(chrom, str):
        chrom = chrom if chrom.startswith("chr") else "chr{}".format(chrom)
    else:
        chrom = [chrom if chrom.startswith(
            "chr") else "chr{}".format(chrom) for c in chrom]
    return chrom


def ensembl_remove_version(ensembl_id: str) -> str:
    if ensembl_id.startswith("ENS"):
        suffix = ""
        if ensembl_id.endswith("_PAR_Y"):
            suffix = "_PAR_Y"
        ensembl_id = ensembl_id.split('.')[0] + suffix
    return ensembl_id


def fix_peak_name(name: str, sep: str) -> str:
    c, s, e = name.split(sep)
    return "{}:{}-{}".format(c, s, e)


## gene & peaks
def load_gene_info(tss_bed, prefix=None) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Return
    -------
    gene2chrom : Dict[str, List[str]], {"ENSGxxx": ["chr1", "chr2", ...], ...}
    gene2tss : Dict[str, Tuple[str, int]], {"ENSGxxx": [["chr1": 100], ["chr1": 102], ...], ...}
    gene_id2name : Dict[str, List[str]], 
    gene_name2id :
    """
    all_loaded = True if prefix is not None else False
    if prefix is not None:
        try:
            gene2chrom = json.load(open("{}.gene2chrom.json".format(prefix)))
            gene2tss = json.load(open("{}.gene2tss.json".format(prefix)))
            gene_id2name = json.load(
                open("{}.gene_id2name.json".format(prefix)))
            gene_name2id = json.load(
                open("{}.gene_name2id.json".format(prefix)))
            logger.info("- logging cache from {}.*.json".format(prefix))
        except FileNotFoundError:
            all_loaded = False

    if not all_loaded:
        gene2tss, gene2chrom = defaultdict(set), defaultdict(set)
        gene_name2id, gene_id2name = defaultdict(set), defaultdict(set)
        non_chrom_contigs = set()
        with open(tss_bed) as infile:
            for l in infile:
                chrom, _, tss, name, _, strand = l.strip().split('\t')
                non_chrom_contigs.add(chrom)
                # if chrom not in HUMAN_CHROMS_NO_MT:
                #     continue
                gene_id, gene_name, gene_type, tx_id, tss, strand = name.split(
                    '|')
                gene_id = ensembl_remove_version(gene_id)
                tss = int(tss)
                gene2tss[gene_id].add((chrom, tss, strand))
                gene2tss[gene_name].add((chrom, tss, strand))
                gene2chrom[gene_id].add(chrom)
                gene2chrom[gene_name].add(chrom)

                gene_name2id[gene_name].add(gene_id)
                gene_id2name[gene_id].add(gene_name)
        non_chrom_contigs = non_chrom_contigs.difference(HUMAN_CHROMS_ALL)

        gene2chrom = dict(gene2chrom)
        gene2tss = dict(gene2tss)
        gene_name2id = dict(gene_name2id)
        gene_id2name = dict(gene_id2name)

        for g, chroms in gene2chrom.items():
            if len(chroms) > 1 and 'chrY' in chroms:
                chroms.remove('chrY')
            if len(chroms.difference(non_chrom_contigs)) > 0:
                chroms = chroms.difference(non_chrom_contigs)
            tss = list()
            for c, t, strand in gene2tss[g]:
                if c in chroms:
                    tss.append([c, t, strand])
            gene2tss[g] = tss
            gene2chrom[g] = list(chroms)

        gene_name2id = {g: list(v) for g, v in gene_name2id.items()}
        gene_id2name = {g: list(v) for g, v in gene_id2name.items()}

        if prefix is not None:
            json.dump(gene2chrom, open(
                "{}.gene2chrom.json".format(prefix), 'w'), indent=4)
            json.dump(gene2tss, open(
                "{}.gene2tss.json".format(prefix), 'w'), indent=4)
            json.dump(gene_id2name, open(
                "{}.gene_id2name.json".format(prefix), 'w'), indent=4)
            json.dump(gene_name2id, open(
                "{}.gene_name2id.json".format(prefix), 'w'), indent=4)
    return gene2chrom, gene2tss, gene_id2name, gene_name2id


def load_fasta(fn: str, no_chr: bool = False, ordered: bool = False, cache: bool = True, gencode_style: bool = False) -> Dict[str, str]:
    r"""
    load fasta as sequence dict
    Input
    -----
    fn : path to fasta file
    ordered : False - dict, True - OrderedDict
    gencode_style : seq_id|xxxxxx

    Return
    -------
    seq_dict : Dict[str, str] or OrderedDict[str, str]
    """
    # if fn == "GRCh38":
    #     fn = HG38_FASTA
    #     logger.warning("- using {}".format(fn))
    #     cache = True
    # elif fn == "GRCh37":
    #     fn = HG19_FASTA
    #     logger.warning("- using {}".format(fn))
    #     cache = True

    if ordered:
        fasta = OrderedDict()
    else:
        fasta = dict()
    name, seq = None, list()
    if cache:
        if no_chr:
            cache = fn + \
                (".gencode.nochr.cache.pkl" if gencode_style else ".nochr.cache.pkl")
        else:
            cache = fn + \
                (".gencode.cache.pkl" if gencode_style else ".cache.pkl")
    else:
        cache = None
    if cache is not None and os.path.exists(cache):
        # logger.info("- load processed genome: {}".format(cache))
        logger.warning("- load processed genome: {}".format(cache))
        fasta = pickle.load(open(cache, 'rb'))
    else:
        with copen(fn) as infile:
            for l in infile:
                if l.startswith('>'):
                    if name is not None:
                        # print("{}\n{}".format(name, ''.join(seq)))
                        if no_chr:
                            name = name.replace("chr", '')
                        fasta[name] = ''.join(seq)
                    if gencode_style:
                        name = l.strip().lstrip('>').split('|')[0]
                    else:
                        name = l.strip().lstrip('>').split()[0]
                    seq = list()
                else:
                    seq.append(l.strip())
        fasta[name] = ''.join(seq)
        if cache is not None:
            try:
                pickle.dump(fasta, open(cache, 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)
            except IOError as err:
                warnings.warn("{}".format(err))
    return fasta


def get_reverse_strand(seq, join: bool = True, integer: bool = False):
    if integer:
        seq = NN_COMPLEMENT_INT[seq][::-1].copy()
    else:
        if join:
            seq = ''.join([NN_COMPLEMENT.get(n, n) for n in seq[::-1]])
        else:
            seq = [NN_COMPLEMENT.get(n, n) for n in seq[::-1]]
    return seq

_NN_ONEHOT = np.concatenate((
    np.ones((1, 4)) / 4,
    np.diag(np.ones(4))
), dtype=np.float16)

class Hdf5Genome(object):
    def __init__(self, fasta) -> None:
        if not fasta.endswith(".h5") and not fasta.endswith(".hdf5"):
            if fasta.endswith(".fa.gz"):
                bn = fasta.replace(".fa.gz", "")
            elif fasta.endswith(".fa"):
                bn = fasta.replace(".fa", "")
            if os.path.exists(bn + ".h5"):
                fasta = bn + ".h5"
            elif os.path.exists(bn + ".fa.h5"):
                fasta = bn + ".fa.h5"
            elif os.path.exists(bn + ".hdf5"):
                fasta = bn + ".hdf5"
            else:
                raise FileNotFoundError("missing hdf5 file {}".format(fasta))
        assert os.path.exists(fasta)
        self.genome = h5py.File(fasta, 'r')
        self.number2text = np.asarray(
            ['N', 'A', 'C', 'G', 'T', 't', 'g', 'c', 'a'])
        # 0

    def fetch(self, chrom: str, start: int, end: int, text=True, reverse: bool = False, onehot: bool=False) -> str:
        if onehot:
            text = False
        seq = self.genome[chrom][start:end]
        if text:
            seq = ''.join(self.number2text[seq])
        if reverse:
            seq = get_reverse_strand(seq, integer=not text)
        if onehot:
            seq = _NN_ONEHOT[seq]
        return seq

def pad_sequences(seq: np.ndarray, seq_start, seq_end, boundary_start, boundary_end, pad_ar):
    pass


def _counts_per_size(mtx: np.ndarray, log: bool = False, target_reads: int = 1e6) -> np.ndarray:
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


def counts_per_thousand(mtx, log: bool = False) -> np.ndarray:
    return _counts_per_size(mtx, log, target_reads=1000)


def counts_per_million(mtx, log: bool = False) -> np.ndarray:
    return _counts_per_size(mtx, log, target_reads=1E6)


def _reverse_counts_per_scale(mtx, libsize, target_reads) -> np.ndarray:
    raise NotImplementedError


def reverse_counts_per_million():
    raise NotImplementedError
    return _reverse_counts_per_scale()


class Chrom2Int(object):
    def __init__(self) -> None:
        self.mapping = _CHROM2INT.copy()
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
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


def load_chrom_size(fai) -> Dict[str, int]:
    chrom_size = dict()
    with open(fai) as infile:
        for l in infile:
            chrom, size = l.split()[:2]
            chrom_size[chrom] = int(size)
    # if chrom.startswith("chr"):
    #     chrom_size[chrom[3:]] = int(size)
    return chrom_size
