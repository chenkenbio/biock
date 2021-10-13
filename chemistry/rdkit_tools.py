#!/usr/bin/env python3

import rdkit
import warnings
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
import hashlib


def smiles2mol(smiles):
    return Chem.MolFromSmiles(smiles)

def get_morgan_fp(mol, radius=2, bits=1024, to_md5=False):
    if isinstance(mol, str):
        smiles = mol
        mol = smiles2mol(smiles)
    else:
        smiles = Chem.MolToSmiles(mol)
    # try:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
    if to_md5:
        fp = ''.join([str(b) for b in np.array(fp).astype(int).reshape(-1)])
        fp = str(hashlib.md5(fp.encode()).hexdigest())
    # except:
    #     warnings.warn("Failed for smiles: {}".format(smiles))
    #     fp = None
    return fp

def mol_dice_similarity(m1, m2):
    if isinstance(m1, str):
        m1 = smiles2mol(m1)
    if isinstance(m2, str):
        m2 = smiles2mol(m2)
    return DataStructs.DiceSimilarity(get_morgan_fp(m1), get_morgan_fp(m2))

