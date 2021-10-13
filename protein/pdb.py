#!/usr/bin/env python3

import os, warnings, logging
import numpy as np
from typing import Dict, OrderedDict
from biock.biock import custom_open
from biock.protein import AA321


class DSSP(object):
    def __init__(self, fn) -> None:
        super().__init__()
        self.residues = dict()
        self.__read_dssp(fn)
    
    def __read_dssp(self, fn):
        start = False
        chain = None
        with open(fn) as infile:
            for l in infile:
                if l.startswith("  #  RESIDUE AA"):
                    assert l[5:12] == "RESIDUE"
                    assert l[13:15] == "AA"
                    assert l[16:25] == "STRUCTURE"
                    assert l[35:38] == "ACC"
                    start = True
                elif start:
                    if l[13:15] == "!*":
                        chain = None
                        continue
                    elif l[13:15] == "! ":
                        continue
                    # if chain is None:
                    chain = l[11]
                    if chain not in self.residues:
                        self.residues[chain] = OrderedDict()
                    # else:
                    #     assert chain == l[11], "{}\n{}\n{}\t{}".format(fn, l, chain, l[11])
                    aa = l[13]
                    if aa.islower():
                        aa = 'C'
                    ss = l[16]
                    acc = int(l[34:38].strip())
                    resSeq = "{}{}".format(int(l[6:10].strip()), l[10])
                    self.residues[chain][resSeq] = (
                        aa, ss, acc
                    )


class PDB(object):
    def __init__(self, filename):
        super().__init__()
        self.resolution = "NONE"
        self.chains = OrderedDict()
        self.resSeq = OrderedDict()
        self.coordinates = OrderedDict()
        self.resSeq2idx = OrderedDict() # (chain, resSeq) -> idx
        self.process(filename)
    
    def process(self, filename):
        last_chain, last_resSeq = None, -1
        terminated = set()
        with custom_open(filename) as infile:
            for l in infile:
                if l.startswith("HEADER"):
                    self.pdb_id = l.strip().split()[-1].lower()
                elif l.startswith("REMARK   2") and len(l.strip()) > 12:
                    try:
                        self.resolution = float(l[23:30].strip())
                        self.resolution = l[23:30].strip()
                        if l[31:41] != "ANGSTROMS.":
                            warnings.warn("Undefined unit: {}".format(l))
                            self.resolution = "{}{}".format(self.resolution, l[31:41].strip())
                    except:
                        pass
                        logging.debug("resolution is not available in {}".format(filename))
                elif l.startswith("ATOM") or l.startswith("HETATM"):
                    chain, resSeq, achar = l[21], int(l[22:26].strip()), l[26]
                    assert last_chain is None or last_chain == chain
                    if chain in terminated:
                        continue
                    if l.startswith("ATOM") and l[12:16].strip() != "CA":
                        continue
                    if l.startswith("HETATM") and resSeq <= max(last_resSeq, 0):
                        continue
                    last_resSeq = resSeq
                    if chain not in self.chains:
                        self.chains[chain] = list()
                        self.resSeq[chain] = list()
                        self.coordinates[chain] = list()
                        self.resSeq2idx[chain] = dict() # int -> int
                    self.resSeq[chain].append("{}{}".format(resSeq, achar))
                    coords = np.array([
                        float(l[30:38].strip()), 
                        float(l[38:46].strip()), 
                        float(l[46:54].strip())
                    ], dtype=np.float16).reshape(1, -1)
                    self.coordinates[chain].append(coords)
                    if l.startswith("HETATM"):
                        aa = 'X'
                    else:
                        aa = AA321[l[17:20]]
                    self.chains[chain].append(aa)
                    self.resSeq2idx[chain]["{}{}".format(resSeq, achar)] = len(self.chains[chain]) - 1
                elif l.startswith("TER "):
                    terminated.add(chain)
                    last_chain, last_resSeq = None, -1
                    del chain
                elif l.startswith("ENDMDL"):
                    break
        chains = list(self.chains.keys())
        for c in chains:
            if len(set(self.chains[c]).difference({'X'})) == 0:
                del self.chains[c], self.coordinates[c], self.resSeq[c], self.resSeq2idx[c]
            else:
                self.chains[c] = ''.join(self.chains[c])
                self.resSeq[c] = np.array(self.resSeq[c])
                self.coordinates[c] = np.concatenate(self.coordinates[c], axis=0)
    
    def print_chains(self):
        for chain, seq in self.chains.items():
            print(">{}{}|RESO={}\n{}".format(self.pdb_id.lower(), chain, "{:.1f}", ''.join(seq)))


def fetch_dssp_features(dssp: DSSP, pdb: PDB, chain: str=None) -> Dict:
    dssp_features = dict()
    for chain in pdb.chains:
        dssp_features[chain] = dict()
        for idx, resSeq in enumerate(pdb.resSeq[chain]):
            if resSeq in dssp.residues[chain]:
                dssp_features[chain][idx] = dssp.residues[chain][resSeq]
    return dssp_features

