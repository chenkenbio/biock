#!/usr/bin/env python3
"""
Load JASPAR motifs as CNN weights

Author: Ken Chen (chenkenbio@gmail.com)
Date: <<date>>
"""

import argparse, os, sys, time
import torch
import torch.nn as nn


JASPAR_DB = "/data2/users/chenken/others/liudq/hic/data/JASPAR2020_CORE_vertebrates_non-redundant_pfms_meme.txt"


class JASPARmotifs(object):
    """Docstring for JASPARmotifs. """
    def __init__(self, jaspar_meme=JASPAR_DB):
        self.jaspar_meme = jaspar_meme
        self.motifs = dict()
        self.motif_ids = list()
        self.motif_names = list()
        self.__process()

    def jaspar_cnn(self, motifs: dict) -> torch.nn.Conv1d:
        max_len = max([self.motifs[m].size(2) for m in motifs])
        w = torch.cat([self.__expand_filter(self.motifs[m], max_len) for m in motifs], dim=0)
        conv = nn.Conv1d(in_channels=4, out_channels=len(motifs), kernel_size=max_len, bias=False)
        conv.load_state_dict({'weight': w})
        return conv

    def __process(self):
        motif_id, motif_name = None, None
        w = list()
        with open(self.jaspar_meme) as infile:
            for l in infile:
                if l.startswith('MOTIF MA'):
                    motif_id, motif_name = l.strip().split(' ')[1:3]
                    self.motif_ids.append(motif_id)
                    self.motif_names.append(motif_name)
                elif l.startswith('URL http'):
                    self.motifs[motif_id] = torch.cat((w), dim=1).unsqueeze(0)
                    self.motifs[motif_name] = torch.cat((w), dim=1).unsqueeze(0)
                    w = list()
                    motif_id, motif_name = None, None
                elif l.startswith(' ') and motif_id is not None:
                    w.append(
                            torch.as_tensor(
                                [float(x) for x in l.strip().split('  ')], dtype=torch.float
                            ).view(4, 1)
                        )
    
    def __expand_filter(self, t: torch.Tensor, max_len: int) -> torch.Tensor:
        print(t.size())
        if t.size(2) < max_len:
            before = torch.zeros((1, 4, (max_len - t.size(2)) // 2))
            print("before", before.size())
            after = torch.zeros((1, 4, max_len - t.size(2) - before.size(2)))
            print("after", after.size())
            t = torch.cat((before, t, after), dim=2)
        print(t.size())
        return t


