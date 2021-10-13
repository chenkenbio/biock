#!/usr/bin/env python3

import argparse, os, sys, time, shutil
# import logging, warnings, json, gzip, pickle
# warning = logging.warning


from collections import defaultdict, OrderedDict
from Bio.PDB.PDBParser import PDBParser

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from scipy.stats import pearsonr, spearmanr
from typing import Any, Dict, List, Union
from functools import partial
from biock.biock import custom_open, run_bash
print = partial(print, flush=True)


__aa3 = ["ALA","CYS","ASP","GLU","PHE","GLY","HIS","ILE","LYS","LEU",
		"MET","ASN","PRO","GLN","ARG","SER","THR","VAL","TRP","TYR", "SEC", "PYL", "UNK", "ASX", "GLX", "XLE", "MSE"]
__aa1 = list("ACDEFGHIKLMNPQRSTVWYUOXBZJX")

AA321 = {k:v for k, v in list(zip(__aa3, __aa1))}

# class PDB(object):
#     def __init__(self, pdb_id: str, filename):
#         self.id = pdb_id.lower()
#         self.resolution = None
#         self.chains = OrderedDict()
#         self.process(filename)
# 
#     def process(self, filename):
#         if filename.endswith("gz"):
#             shutil.copy2(filename, "/tmp/{}".format(os.path.basename(filename)))
#             filename = "/tmp/{}".format(os.path.basename(filename))
#             rc, out, err = run_bash("gzip -df {}".format(filename))
#             if rc != 0:
#                 raise RuntimeError(err)
#             filename = filename.replace('.gz', '')
#         parser = PDBParser()
#         struct = parser.get_structure(self.id, filename)
#         header = struct.header.copy()
#         self.resolution = header["resolution"]
# 
#         for chain in struct.get_chains():
#             chain_id = chain.get_id()
#             try:
#                 seq = ''.join([AA321[r.get_resname()] for r in chain.get_residues()])
#             except KeyError as err:
#                 print(' '.join([r.get_resname() for r in chain.get_residues()]))
#                 raise RuntimeError(err)
#             self.chains[chain_id] = {"chain": seq}
# 
#     def print_chains():
#         for chain in self.chains:
#             seq = self.chains[chain]["chain"]
#             print(">{}{}|resolution={}\n{}".format(
#                     self.id.lower(), 
#                     chain.upper(),
#                     self.resolution,
#                     seq
#                 ))
# class PDB(object):
#     def __init__


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #p.add_argument()

    #p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    #np.random.seed(args.seed)

