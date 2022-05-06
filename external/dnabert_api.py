#!/usr/bin/env python3

## configuration
BASE_DIR = "/home/chenken/biock/biock/external/DNABERT"
PATH_TO_DNABERT_SRC = "{}/src".format(BASE_DIR)
PATH_TO_DNABERT_MODEL_DIR = {
    '6': "{}/models/6-new-12w-0".format(BASE_DIR),
    6: "{}/models/6-new-12w-0".format(BASE_DIR),
    '5': "{}/models/5-new-12w-0".format(BASE_DIR),
    5: "{}/models/5-new-12w-0".format(BASE_DIR)
}

import argparse
import pickle
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
import os
import sys
from biock.gpu import select_device
dev_id, dev = select_device()
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from biock import load_fasta, copen
import logging
logger = logging.getLogger(__name__)

sys.path.append(PATH_TO_DNABERT_SRC)
from dnabert_transformers import BertModel, BertConfig, DNATokenizer

def generate_kmer(seq: str, k: int):
    seq = seq.upper()
    kmer = list()
    for i in range(len(seq) - k + 1):
        kmer.append(seq[i:i + k])
    return ' '.join(kmer)


class SeqData(Dataset):
    r"""
    load DNA sequences from fasta file or sequence list

    Input
    -----
    sequence : Union[str, List[Tuple[str, str]]] : fasta file or sequence list ([["seq1", "AGCT"], ...])
    """
    def __init__(self, sequence: Union[str, List[Tuple[str, str]]], genome_dict, k: int) -> None:
        super().__init__()
        self.k = k
        self.tokenizer = DNATokenizer.from_pretrained('dna{}'.format(k))

        self.genome_dict = pickle.load(open(genome_dict, 'rb'))
        self.seqs = list()
        if type(sequence) is str:
            d = load_fasta(sequence, ordered=True)
            for n, s in d.items():
                self.seqs((n, s))
        else:
            for n, s in sequence:
                self.seqs.append((n, s))
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, index):
        seq = self.seqs[index][1]
        seq = generate_kmer(seq, k=self.k)
        seq = self.tokenizer.encode_plus(seq, add_special_tokens=True, max_length=512)["input_ids"]
        seq = torch.as_tensor(seq, dtype=torch.long)
        return seq

    def collate_fn(self, seqs: List[Tensor]):
        lengths = [x.shape[0] for x in seqs]
        max_size = max(lengths)
        mask = torch.stack([torch.cat((torch.ones(s), torch.zeros(max_size - s))) for s in lengths])
        seqs = pad_sequence(seqs).transpose(0, 1)
        return seqs, mask


class BedData(Dataset):
    r"""
    load sequences based on bed file
    """
    def __init__(self, bed, genome_dict, k: int, length: int=None):
        super(BedData, self).__init__()
        self.length = length
        self.k = k
        self.tokenizer = DNATokenizer.from_pretrained('dna{}'.format(k))

        self.seqs = list()

        if type(bed) is str:
            self.load_sequence_from_bed(bed)
        else:
            self.load_sequence_from_list(bed)
        self.genome_dict = pickle.load(open(genome_dict, 'rb'))
       
    def __getitem__(self, index):
        chrom, start, end, _, _ = self.seqs[index]
        pad_left, pad_right = str(), str()
        if start < 0:
            pad_left = 'N' * (-start)
            start = 0
        if end > len(self.genome_dict[chrom]):
            pad_right = 'N' * (len(self.genome_dict[chrom]) - end)
            end = len(self.genome_dict[chrom])
        seq = self.genome_dict[chrom][start:end]
        seq = ''.join([pad_left, seq, pad_right])
        seq = generate_kmer(seq, k=self.k)
        seq = self.tokenizer.encode_plus(seq, add_special_tokens=True, max_length=512)["input_ids"]
        seq = torch.as_tensor(seq, dtype=torch.long)
        return seq
    
    def __len__(self):
        return len(self.seqs)

    def load_sequence_from_list(self, peak_list):
        """
        peak_list: ("chr1", 100, 200)
        """
        for peak in peak_list:
            chrom, start, end = peak[0], int(peak[1]), int(peak[2])
            name = "{}:{}-{}".format(*peak[0:3])
            if self.length is not None:
                mid = (start + end) // 2
                start, end = mid - self.length // 2, mid + self.length // 2
            self.seqs.append((chrom, start, end, name, peak))
 
    def load_sequence_from_bed(self, bed):
        with open(bed) as infile:
            for l in infile:
                fields = l.rstrip('\n').split('\t')
                if len(fields) == 3:
                    name = "{}:{}-{}".format(*fields[0:3])
                else:
                    name = fields[4]
                chrom, start, end = fields[0], int(fields[1]), int(fields[2])
                if self.length is not None:
                    mid = (start + end) // 2
                    start, end = mid - self.length // 2, mid + self.length // 2
                self.seqs.append((chrom, start, end, name, l.rstrip('\n')))
    
    def collate_fn(self, seqs: List[Tensor]):
        lengths = [x.shape[0] for x in seqs]
        max_size = max(lengths)
        mask = torch.stack([torch.cat((torch.ones(s), torch.zeros(max_size - s))) for s in lengths])
        seqs = pad_sequence(seqs).transpose(0, 1)
        return seqs, mask

        
def load_model(k: int):
    dir_to_pretrained_model=PATH_TO_DNABERT_MODEL_DIR[k]
    model = BertModel.from_pretrained("{}".format(dir_to_pretrained_model))
    model.eval()
    return model

def get_embedding_from_fasta(fasta, genome_dict: str, k: int=6, batch_size: int=128, device=torch.device("cuda"), verbose=False, fragment_only: bool=False) -> Tuple[List[str], np.ndarray, np.ndarray]:
    r"""
    Input
    -----
    fasta: fasta file or sequences
    genome_dict : genome dict in pickle format
    k : k-mer, 5 or 6
    batch_size : batch size
    device : 

    Return
    -------
    name : name list
    basewise_embedding : (B, S, E)
    fragment_embedding : (B, E)

    """

    model = load_model(k=k).to(device)
    basewise_embedding = list()
    fragment_embedding = list()
    data = SeqData(fasta, genome_dict=genome_dict, k=k)
    if verbose:
        logger.info("- load {} sequences".format(len(data.seqs)))
    loader = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, num_workers=max(0, min(batch_size, 16) - 1))
    names = [data.seqs[i][0] for i in range(len(data))]

    with torch.no_grad():
        model.eval()
        for seq, mask in tqdm(loader, total=len(loader), desc="DNABERT predicting"):
            seq, mask = seq.to(device), mask.to(device)
            base, frag = model.forward(input_ids=seq, attention_mask=mask)
            if not fragment_only:
                basewise_embedding.append(base.detach().cpu().numpy())
            fragment_embedding.append(frag.detach().cpu().numpy())
    del model, base, frag, seq, mask
    if not fragment_only:
        basewise_embedding = np.concatenate(basewise_embedding)
    else:
        basewise_embedding = np.array([])
    fragment_embedding = np.concatenate(fragment_embedding)
    return names, basewise_embedding, fragment_embedding

def get_embedding_from_bed(bed: Union[str, List], genome_dict, k: int=6, batch_size: int=128, length: int=None, device=torch.device("cuda"), verbose=False) -> Tuple[List[str], np.ndarray, np.ndarray]:
    r"""
    Input
    -----
    bed : bed file
    genome_dict : genome dict in pickle format
    k : k-mer, 5 or 6
    batch_size : batch size
    length : whether use uniform length
    device : 

    Return
    -------
    name : name list
    basewise_embedding : (B, S, E)
    fragment_embedding : (B, E)

    """

    model = load_model(k=k).to(device)
    basewise_embedding = list()
    fragment_embedding = list()
    data = BedData(bed, genome_dict=genome_dict, length=length, k=k)
    if verbose:
        logger.info("- load {} sequences".format(len(data.seqs)))
    loader = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, num_workers=max(0, min(batch_size, 16) - 1))
    names = [data.seqs[i][3] for i in range(len(data))]

    with torch.no_grad():
        model.eval()
        for seq, mask in tqdm(loader, total=len(loader), desc="DNABERT predicting"):
            seq, mask = seq.to(device), mask.to(device)
            base, frag = model.forward(input_ids=seq, attention_mask=mask)
            basewise_embedding.append(base.detach().cpu().numpy().astype(np.float16))
            fragment_embedding.append(frag.detach().cpu().numpy().astype(np.float16))
    del model, base, frag, seq, mask
    basewise_embedding = np.concatenate(basewise_embedding)
    fragment_embedding = np.concatenate(fragment_embedding)
    return names, basewise_embedding, fragment_embedding
