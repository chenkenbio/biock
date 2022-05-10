#!/usr/bin/env python3
"""
Author: Ken Chen (chenkenbio@gmail.com)
Date: <<date>>
"""

from typing import Any, List, Optional, Union, Iterable, Literal, Dict
import scanpy as sc
from anndata import AnnData
import scipy.sparse as ssp
import pandas as pd
from pandas import Index
from sklearn.feature_extraction.text import TfidfTransformer
import gzip
from scipy.sparse import csr_matrix, issparse
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

def random_adata(nrow: int, ncol: int, density: float=0.001, dtype=np.float32) -> AnnData:
    return AnnData(X=ssp.random(nrow, ncol, density=density, format="csr", dtype=dtype), dtype=dtype)


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

def fix_peak_index(peaks: Iterable[str]) -> Index:
    r"""
    Parameters
    -----------
    peaks : AnnData.var.index
        (["chr1_100_200", ...]/["chr1\t100\t200", ...]/["chr1:100-200", ...])
    
    Return
    -------
    peaks : peak names in standard format (chr1:100-200)
    """
    p = peaks[0]
    if '\t' in p:
        sep = '\t'
    elif '_' in p:
        sep = '_'
    else:
        sep = None
    if sep is not None:
        new_peaks = list()
        for p in peaks:
            c, s, e = p.split(sep)
            new_peaks.append("{}:{}-{}".format(c, s, e))
        return new_peaks
    else:
        return peaks

def tfidf_transform(
        sparse_matrix: csr_matrix, 
        norm: Literal["l2", "l1", None]=None, 
        binary :bool=True, 
        get_model: bool=False) -> csr_matrix:
    if binary:
        sparse_matrix.data = np.ones_like(sparse_matrix.data, dtype=np.float32)
    tfidf = TfidfTransformer(norm=norm)
    sparse_matrix = tfidf.fit_transform(sparse_matrix)
    if get_model:
        return sparse_matrix, tfidf
    else:
        return sparse_matrix


def stat_adata(adata: AnnData) -> Dict[str, Any]:
    if isinstance(adata.X, np.ndarray):
        d = {
            "shape": adata.shape,
            "X-min/mean/max": (adata.X.min(), adata.X.mean(), adata.X.max()),
        }
    else:
        d = {
            "shape": adata.shape,
            "density": round(len(adata.X.data) / np.prod(adata.X.shape), 6),
            "X-min/max": (adata.X.min(), adata.X.max()),
            "X-min/mean/max(nonzero)": (round(adata.X.data.min(), 4), round(adata.X.data.mean(), 4), round(adata.X.data.max(), 4))
        }
    return d
