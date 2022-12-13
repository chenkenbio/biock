#!/usr/bin/env python3
"""
Author: Ken Chen
Email: chenkenbio@gmail.com
Date: 2022-12-11
"""

import argparse
import os
import sys
import numpy as np
import warnings

from biock import auto_open
import vcf

def none2str(s):
    if s is None:
        s = '.'
    return s

class VCFData(object):
    def __init__(self, vcf, keep_record=True) -> None:
        super().__init__()
        self.vcf = vcf
        self.keep_record = keep_record
        self.records = list()
        self.var_inds = list()
        self.chroms = list()
        self.starts = list()
        self.ends = list()
        self.refs = list()
        self.alts = list()
        # self.strands = list()

        self.names = list() # may be less than chroms/positions/refs/alts
        self.infos = list()

        self.process()
    
    def process(self):
        use_pyvcf = True
        try:
            import vcf
        except ImportError as err:
            use_pyvcf = False
            warnings.warn("missing 'vcf' library (pyvcf), use custom parser")
        if use_pyvcf:
            vcfin = vcf.Reader(auto_open(self.vcf))
            for idx, variant in enumerate(vcfin):
                self.names.append(variant.ID)
                self.infos.append(variant.INFO)
                record = '_'.join([
                            variant.CHROM,
                            str(variant.POS),
                            variant.REF,
                            ','.join([x.sequence for x in variant.ALT]),
                        ])

                for alt in variant.ALT:
                    alt = alt.sequence
                    # strand = list(set(variant.INFO['strand']))
                    # if len(strand) > 1 or strand[0] == '.':
                        # strand = '.'
                        # warnings.warn("ambiguous strand info in {} {}".format(variant, variant.INFO["strand"]))
                    # else:
                        # strand = strand[0]
                    self.var_inds.append(idx)
                    self.chroms.append(variant.CHROM)
                    if self.keep_record:
                        self.records.append(record)
                    # self.strands.append(strand)
                    if len(variant.REF) > 1 or len(alt) > 1:
                        shift = 0
                        while variant.REF[shift] == alt[shift]:
                            shift += 1
                            if shift >= len(variant.REF) or shift >= len(alt):
                                break
                        self.refs.append(variant.REF[shift:])
                        self.alts.append(alt[shift:])
                        start = variant.POS - 1 + shift
                        end = start + len(self.refs[-1])
                        self.starts.append(start)
                        self.ends.append(end)
                    else:
                        self.starts.append(variant.POS - 1)
                        self.ends.append(variant.POS - 1 + len(variant.REF))
                        self.refs.append(variant.REF)
                        self.alts.append(alt)
        else:
            raise NotImplementedError
            with auto_open(self.vcf) as infile:
                for l in infile:
                    if l.startswith("#"):
                        pass
        self.chroms = np.asarray(self.chroms)
        self.starts = np.asarray(self.starts)
        self.ends = np.asarray(self.ends)
        self.refs = np.asarray(self.refs)
        self.alts = np.asarray(self.alts)
        # self.strands = np.asarray(self.strands)
    
    def to_bed(self):
        bed_list = list()
        for chrom, start, end, ref, alt, var_id in zip(self.chroms, self.starts, self.ends, self.refs, self.alts, self.var_inds):
            name = self.records[var_id]
            var_info = self.infos[var_id]
            strand = var_info.get("strand", ['.'])
            if len(strand) > 1 or strand[0] == '.':
                strand = '.'
            else:
                strand = strand[0]

            bed_list.append((
                chrom, start, end, name, '.', strand
            ))
        return bed_list

