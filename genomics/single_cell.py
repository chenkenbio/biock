#!/usr/bin/env python3
"""
Author: Ken Chen (chenkenbio@gmail.com)
Date: <<date>>
"""

from typing import List, Optional
import scanpy as sc
from anndata import AnnData
import scipy.sparse as ssp
import pandas as pd
import gzip
from scipy.sparse import csr_matrix
import numpy as np

from ..logger import make_logger
logger = make_logger()

def keep_common_cells(*args: AnnData) -> List[AnnData]:
    for i in range(len(args)):
        if i == 0:
            index = set(args[i].obs.index)
            print(len(index))
        else:
            index = index.intersection(set(args[i].obs.index))
    index = sorted(list(index))
    print(len(index))
    return [a[index, ] for a in args]

def random_adata(nrow: int, ncol: int, density: float=0.001) -> AnnData:
    return AnnData(X=ssp.random(nrow, ncol, density=0.001, format="csr", dtype=np.float32))


## convert anndata to mtx.gz, features.tsv.gz and barcodes.tsv.gz
def _sparse_matrix_to_mtx(mtx: csr_matrix, fn: str):
    row, col = mtx.nonzero()
    data = mtx.data
    logger.info("- shape/min/max: {}".format(data.shape, data.min(), data.max()))
    assert len(row) == len(data)
    with gzip.open(fn, 'wt') as out:
        out.write("%%MatrixMarket matrix coordinate real general\n%%\n")
        max_col, max_row, n_nonzero = max(col) + 1, max(row) + 1, len(col)
        out.write("{} {} {}\n".format(max_col, max_row, n_nonzero))
        # for c, r, v in tqdm(zip(col, row, data), total=len(row)):
        for c, r, v in zip(col, row, data):
            out.write("{} {} {}\n".format(c + 1, r + 1, int(v)))
    logger.info('done')

def _save_as_mtx(matrix: csr_matrix, barcodes: np.ndarray, features: np.ndarray, prefix: str, obs: Optional[pd.DataFrame]=None, var: Optional[pd.DataFrame]=None):
    """
    convert 
    """
    _sparse_matrix_to_mtx(matrix, "{}.mtx.gz".format(prefix))
    np.savetxt("{}.barcodes.tsv.gz".format(prefix), barcodes, fmt="%s")
    np.savetxt("{}.features.tsv.gz".format(prefix), features, fmt="%s")
    if obs is not None:
        obs.to_csv("{}.obs.tsv".format(prefix), sep='\t')
    if var is not None:
        var.to_csv("{}.var.tsv".format(prefix), sep='\t')

def adata2mtx(adata: AnnData, prefix: str, use_rep: str):
    if use_rep != 'X':
        m = adata.layers[use_rep]
    else:
        m = adata.X
    _save_as_mtx(
        m,
        barcodes=adata.obs.index.to_numpy(), 
        features=adata.var.index.to_numpy(), 
        prefix=prefix,
        obs=adata.obs,
        var=adata.var
    )