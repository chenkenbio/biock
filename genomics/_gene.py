#!/usr/bin/env python3
"""
Author: Ken Chen
Email: chenkenbio@gmail.com
Date: 2022-11-17
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

class Transcript(object):

    def __init__(self, tx_id, gene_id, chrom, tx_start, tx_end, strand, cds_start=None, cds_end=None, buildver=None, gene_name=None, **kwargs) -> None:
        self.tx_id = tx_id
        self.gene_id = gene_id
        self.chrom = chrom
        self.tx_start = tx_start
        self.tx_end = tx_end
        self.strand = strand
        self.cds_start = cds_start
        self.cds_end = cds_end
        self.buildver = buildver
        self.gene_name = gene_name
        self._cds: List[Tuple[int, int]]=list()
        self._utr5: List[Tuple[int, int]]=list()
        self._utr3: List[Tuple[int, int]]=list()
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def __repr__(self) -> str:
        repr = list()
        for k, v in [("tx_id", self.tx_id), ("gene_id", self.gene_id), ("gene_name",  self.gene_name), ("chrom", self.chrom), ("tx_start", self.tx_start), ("tx_end", self.tx_end), ("strand", self.strand)]:
            if v is not None:
                repr.append("{}={}".format(k, v))
        return "Transcript({})".format(",".join(repr))
    
    @property
    def intron_starts(self):
        if not hasattr(self, "_intron_starts"):
            self._intron_starts = self.exon_ends[0:-1]
        return self._intron_starts
    @property
    def intron_ends(self):
        if not hasattr(self, "_intron_ends"):
            self._intron_ends = self.exon_starts[1:]
        return self._intron_ends

    @property
    def intron(self):
        return list(zip(self.intron_starts, self.intron_ends))

    @property
    def exon(self):
        return list(zip(self.exon_starts, self.exon_ends))
    
    def _get_utr_cds(self):
        assert self.cds_start is not None and self.cds_end is not None, "the start/end of CDS is unknown"
        if len(self._utr5) == 0 and len(self._cds) == 0 and len(self._utr3) == 0:
            utr_left, utr_right = list(), list()
            for e1, e2 in zip(self.exon_starts, self.exon_ends):
                left, right = max(e1, self.cds_start), min(e2, self.cds_end)
                if left < right:
                    self._cds.append((left, right))
                if e1 < self.cds_start:
                    utr_left.append((e1, min(e2, self.cds_start)))
                if e2 > self.cds_end:
                    utr_right.append((max(self.cds_end, e1), e2))
            if self.strand == '+':
                self._utr5 = utr_left
                self._utr3 = utr_right
            else:
                self._utr5 = utr_right
                self._utr3 = utr_left
    
    def check(self):
        # check cds&utr
        exon_size, cds_size, utr_size = 0, 0, 0
        if self.strand == '+':
            for x, y in self.utr5:
                utr_size += y - x
                assert x < y, "{}".format(((x, y), self.cds_start, self.cds_end))
                assert y <= self.cds_start, "{}".format(((x, y), self.cds_start, self.cds_end))
            for x, y in self.utr3:
                utr_size += y - x
                assert x < y, "{}".format(((x, y), self.cds_start, self.cds_end))
                assert x >= self.cds_end, "{}".format(((x, y), self.cds_start, self.cds_end))
        else:
            for x, y in self.utr5:
                utr_size += y - x
                assert x < y, "{}".format(((x, y), self.cds_start, self.cds_end))
                assert x >= self.cds_end, "{}".format(((x, y), self.cds_start, self.cds_end))
            for x, y in self.utr3:
                utr_size += y - x
                assert x < y, "{}".format(((x, y), self.cds_start, self.cds_end))
                assert y <= self.cds_start, "{}".format(((x, y), self.cds_start, self.cds_end))
        for x, y in self.cds:
            assert x >= self.cds_start, "{}".format(((x, y), self.cds_start, self.cds_end))
            assert y <= self.cds_end, "{}".format(((x, y), self.cds_start, self.cds_end))
            cds_size += y - x
        exon_size = sum([y - x for x, y in zip(self.exon_starts, self.exon_ends)])
        assert exon_size == cds_size + utr_size, "{}".format((exon_size, utr_size, cds_size))
        intervals = sorted(t.utr5 + t.utr3 + t.cds + t.intron, key=lambda x:x[0])
        for i in range(len(intervals) - 1):
            assert intervals[i][1] == intervals[i +  1][0], ';'.join(["({},{})".format(x, y) for x, y in intervals])

    @property
    def cds(self):
        if len(self._cds) == 0:
            self._get_utr_cds()
        return self._cds

    @property
    def utr5(self):
        if len(self._utr5) == 0:
            self._get_utr_cds()
        return self._utr5

    @property
    def utr3(self):
        if len(self._utr3) == 0:
            self._get_utr_cds()
        return self._utr3

    def get_function_region(self, function_region: Literal["exon", "intron", "ss-exon", "ss-intron-1", "ss-intron-2"], name: List[str]='.', name_sep: str='|', numbering: bool=False) -> List[Tuple[str, int, int, str, str, str]]:
        r"""
        Arguments:
        ---
        function_region: 
            exon: 
            intron:
            ss-exon: splice sites in exons (1nt)
            ss-intron-1: splice sites in introns (1nt)
            ss-intron-2: splice sites in introns (2nt)
        name : List[str] feature names to be shown in 'name'
        name_sep : str
        numbering : bool : whether add exon/intron number

        Return:
        ---
        bed_list : (chrom, start, end, name, ., strand)
        """
        if type(name) is str and name != '.':
            name = [name]
        name_prefix = list()
        for k in name:
            if k == '.':
                name_prefix = ""
            elif self.__dict__[k] is None:
                name_prefix.append('NaN')
            else:
                name_prefix.append(self.__dict__[k])
        name_prefix = name_sep.join(name_prefix)

        intervals = list()
        if function_region == "exon":
            for i, (r1, r2) in enumerate(zip(self.exon_starts, self.exon_ends)):
                if numbering:
                    if self.strand == '-':
                        exon_id = "|EXON{}".format(len(self.exon_starts) - i)
                    else:
                        exon_id = "|EXON{}".format(i + 1)
                    iname = exon_id
                else:
                    iname = ""
                intervals.append((r1, r2, "exon{}".format(iname)))
        elif function_region == "intron":
            for i, (r1, r2) in enumerate(zip(self.intron_starts, self.intron_ends)):
                intron_id = ""
                if numbering:
                    if self.strand == '-':
                        intron_id = "|INT{}".format(len(self.intron_starts) - i)
                    else:
                        intron_id = "|INT{}".format(i + 1)
                intervals.append((r1, r2, "intron{}".format(intron_id)))
        elif function_region == "ss-exon":
            for i, (d, a) in enumerate(zip(self.intron_starts, self.intron_ends)):
                if self.strand == '-':
                    intervals.append((d - 1, d, "3'SS"))
                    intervals.append((a, a + 1, "5'SS"))
                else:
                    intervals.append((d - 1, d, "5'SS"))
                    intervals.append((a, a + 1, "3'SS"))
        elif function_region == "ss-intron-1":
            for i, (d, a) in enumerate(zip(self.intron_starts, self.intron_ends)):
                if self.strand == '-':
                    intervals.append((d, d + 1, "acceptor"))
                    intervals.append((a - 1, a, "donor"))
                else:
                    intervals.append((d, d + 1, "donor"))
                    intervals.append((a - 1, a, "acceptor"))
        elif function_region == "ss-intron-2":
            for i, (d, a) in enumerate(zip(self.intron_starts, self.intron_ends)):
                if self.strand == '-':
                    intervals.append((d, d + 2, "acceptor"))
                    intervals.append((a - 2, a, "donor"))
                else:
                    intervals.append((d, d + 2, "donor"))
                    intervals.append((a - 2, a, "acceptor"))
        else:
            raise NotImplementedError("unknown function_region: {}".format(function_region))

        bed_list = list()

        if name_prefix != '':
            name_prefix = name_prefix + name_sep

        for l, r, rname in intervals:
            bed_list.append((self.chrom, l, r, "{}{}".format(name_prefix, rname), '.', self.strand))

        return bed_list

    
    # def get_introns(self, name, name_sep, index):
    #     if type(name) is str:
    #         name = [name]
    #     intron_name = list()
    #     for k in name:
    #         if self.__dict__[k] is None:
    #             intron_name.append('NaN')
    #         else:
    #             intron_name.append(self.__dict__[k])
    #     intron_name = name_sep.join(intron_name)
    #     intron_list = list()
    #     for i, (s, e) in enumerate(zip(self.intron_starts, self.intron_ends)):
    #         if self.strand == '-':
    #             intron_id = "INT{}".format(len(self.intron_starts) - i)
    #         else:
    #             intron_id = "INT{}".format(i + 1)
    #         if index:
    #             iname = intron_name
    #         else:
    #             iname = "{}{}{}".format(intron_name, name_sep, intron_id)
    #         intron_list.append((self.chrom, s, e, iname, '.', self.strand))
    #     return intron_list
    
    
