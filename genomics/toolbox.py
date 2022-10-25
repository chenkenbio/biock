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
from typing import Any, Union, List, Dict, Tuple
import json
import logging
import pickle
from ..utils import copen, has_module
from ..variables import HUMAN_CHROMS_ALL
try:
    import pyfaidx
except ImportError as err:
    warnings.warn("Missing module: pyfaidx\n{}".format(err))

logger = logging.getLogger(__name__)


GTFRecord = namedtuple("GTF_record", field_names=["chrom", "start", "end", "strand", "feature_type", "attrs"])
def parse_gtf_record(gtf_line: str, to_dict=False) -> GTFRecord:
    r"""
    Return:
    -------
    chrom : str
    ftype : str
    start : int
    end : int
    strand : str
    attrs : dict
    """
    try:
        chrom, _, ftype, start, end, _, strand, _, fields = gtf_line.strip().rstrip(';').split('\t')
    except:
        raise RuntimeError("{}".format(gtf_line.strip()))
    attrs = dict()
    for kv in fields.split('; '):
        try:
            v = kv.split(' ')
            k = v[0]
            v = ' '.join(v[1:])
        except ValueError as err:
            print(kv, gtf_line)
            exit(err)
        v = v.strip('"')
        if k in attrs:
            if isinstance(attrs[k], str):
                attrs[k] = set([attrs[k]])
            attrs[k].add(v)
        else:
            attrs[k] = v
    if to_dict:
        return {
                    "chrom":  chrom, 
                    "ftype": ftype, 
                    "start": int(start), 
                    "end": int(end), 
                    "strand": strand, 
                    "attrs": attrs
                }
    else:
        return GTFRecord(chrom=chrom, start=int(start), end=int(end), strand=strand, feature_type=ftype, attrs=attrs)


def chrom_add_chr(chrom: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(chrom, str):
        chrom = "chr{}".format(chrom) if not chrom.startswith("chr") else chrom
    else:
        chrom = ['chr{}'.format(c) if not c.startswith("chr") else c for c in chrom]
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
        chrom = [chrom if chrom.startswith("chr") else "chr{}".format(chrom) for c in chrom]
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
            gene_id2name = json.load(open("{}.gene_id2name.json".format(prefix)))
            gene_name2id = json.load(open("{}.gene_name2id.json".format(prefix)))
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
                gene_id, gene_name, gene_type, tx_id, tss, strand = name.split('|')
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
            json.dump(gene2chrom, open("{}.gene2chrom.json".format(prefix), 'w'), indent=4) 
            json.dump(gene2tss, open("{}.gene2tss.json".format(prefix), 'w'), indent=4)
            json.dump(gene_id2name, open("{}.gene_id2name.json".format(prefix), 'w'), indent=4)
            json.dump(gene_name2id, open("{}.gene_name2id.json".format(prefix), 'w'), indent=4)
    return gene2chrom, gene2tss, gene_id2name, gene_name2id



def load_fasta(fn: str, no_chr: bool=False, ordered: bool=False, cache: bool=True, gencode_style: bool=False) -> Dict[str, str]:
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
            cache = fn + (".gencode.nochr.cache.pkl" if gencode_style else ".nochr.cache.pkl")
        else:
            cache = fn + (".gencode.cache.pkl" if gencode_style else ".cache.pkl")
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
                pickle.dump(fasta, open(cache, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            except IOError as err:
                warnings.warn("{}".format(err))
    return fasta


class Genome(object):
    def __init__(self, fasta, in_memory: bool=True) -> None:
        self.in_memory = in_memory
        if in_memory and has_module("pyfaidx"):
            self.fasta = pyfaidx.Fasta(fasta)
        else:
            self.fasta = load_fasta(fasta)
    
    def get_seq(self, chrom: str, start: int, end: int) -> str:
        seq = self.fasta[chrom][start:end]
        if self.in_memory:
            seq = seq.seq
        return seq
        
        