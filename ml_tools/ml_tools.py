#!/usr/bin/env python3

from typing import Any, Iterable, Tuple, List, Union
from collections import namedtuple
import sklearn.metrics as metrics # auc, precision_recall_curve, roc_auc_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
import warnings
import torch
from torch import Tensor
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

def cyclic_iter(iters):
    while True:
        for x in iters:
            yield x

def roc_auc_score(y_true, y_score, robust: bool=False) -> float:
    if robust and len(np.unique(y_true)) == 1:
        res = np.nan
    else:
        res = metrics.roc_auc_score(y_true, y_score)
    return res


def pr_curve(labels, scores):
    new_labels, new_scores = list(zip(*sorted(zip(labels, scores), key=lambda x:x[1], reverse=True)))
    precision, recall, thresholds = list(), list(), list()
    num_p = sum(new_labels)
    tp, fp = 0, 0
    for i in range(len(new_labels)):
        if new_labels[i] == 1:
            tp += 1
        else:
            fp += 1
        if i == len(new_labels) - 1 or new_scores[i + 1] != new_scores[i]:
            precision.append(tp/(tp + fp))
            recall.append(tp / num_p)
            thresholds.append(new_scores[i])
    return np.asarray(precision), np.asarray(recall), np.asarray(thresholds)


def pr_auc_score(y_true, probas_pred, robust: bool=False):
    if robust and len(np.unique(y_true)) == 1:
        res = np.nan
    else:
        p, r, _ = metrics.precision_recall_curve(y_true, probas_pred)
        res = metrics.auc(r, p)
    return res


def _bce_numpy(y_true: np.ndarray, probas_pred: np.ndarray, eps=1E-15) -> float:
    y_true, probas_pred = np.asarray(y_true), np.asarray(probas_pred)
    assert np.array_equal(y_true.shape, probas_pred.shape), "expected equal size array, but {} and {} were found".format(y_true.shape, probas_pred.shape)
    dtype = (1.0 + y_true[0]).dtype # at least np.float16
    y_true = y_true.astype(dtype)
    probas_pred = probas_pred.astype(dtype)
    probas_pred = np.clip(probas_pred, a_min=eps, a_max=1 - eps)
    bce = -(y_true * np.log(probas_pred) + (1 - y_true) * np.log(1 - probas_pred))
    return np.mean(bce)

def _bce_torch(y_true: Tensor, probas_pred: Tensor, mask: Tensor=None, eps=1E-15) -> Tensor:
    assert np.array_equal(y_true.shape, probas_pred.shape), "expected equal size array, but {} and {} were found".format(y_true.shape, probas_pred.shape)
    dtype = (1.0 + y_true[0]).dtype # at least np.float16
    y_true = torch.as_tensor(y_true, dtype=dtype)
    probas_pred = torch.as_tensor(probas_pred, dtype=dtype).clamp(min=eps, max=1 - eps)
    bce = -(y_true * torch.log(probas_pred) + (1 - y_true) * torch.log(1 - probas_pred))
    if mask is not None:
        bce *= mask
    return torch.mean(bce)

def binary_cross_entropy(y_true: Union[np.ndarray, Tensor], probas_pred: Union[np.ndarray, Tensor], eps: float=1E-15, mask: Tensor=None) -> Union[float, Tensor]:
    if isinstance(y_true, Tensor):
        return _bce_torch(y_true=y_true, probas_pred=probas_pred, eps=eps, mask=mask)
    else:
        return _bce_numpy(y_true=y_true, probas_pred=probas_pred, eps=eps)


def roc_auc_score_2d(y_true, probas_pred, robust: bool=True) -> np.ndarray:
    y_true, probas_pred = np.asarray(y_true), np.asarray(probas_pred)
    assert np.array_equal(y_true.shape, probas_pred.shape), "expected equal size array, but {} and {} were found".format(y_true.shape, probas_pred.shape)
    assert len(y_true.shape) == 2, "2d array is expected"
    auc_list = list()
    for i in range(y_true.shape[0]):
        auc_list.append(roc_auc_score(y_true[i], probas_pred[i], robust=robust))
    return np.asarray(auc_list)


def pr_auc_score_2d(y_true: np.ndarray, probas_pred: np.ndarray, robust: bool=True) -> np.ndarray:
    y_true, probas_pred = np.asarray(y_true), np.asarray(probas_pred)
    assert np.array_equal(y_true.shape, probas_pred.shape), "expected equal size array, but {} and {} were found".format(y_true.shape, probas_pred.shape)
    assert len(y_true.shape) == 2, "2d array is expected"
    auc_list = list()
    for i in range(y_true.shape[0]):
        auc_list.append(pr_auc_score(y_true[i], probas_pred[i], robust=robust))
    return np.asarray(auc_list)


def roc_pr_auc_scores_2d(y_true: np.ndarray, probas_pred: np.ndarray, robust: bool=True) -> Tuple[float, float]:
    """
    Return:
    auc(mean) : float
    auprc(mean) : float
    """
    y_true, probas_pred = np.asarray(y_true), np.asarray(probas_pred)
    assert np.array_equal(y_true.shape, probas_pred.shape), "expected equal size array, but {} and {} were found".format(y_true.shape, probas_pred.shape)
    assert len(y_true.shape) == 2, "2d array is expected, while encountered {}".format(y_true.shape)
    auc_list, auprc_list = list(), list()
    for i in range(y_true.shape[0]):
        auc_list.append(roc_auc_score(y_true[i], probas_pred[i], robust=robust))
        auprc_list.append(pr_auc_score(y_true[i], probas_pred[i], robust=True))
    return np.nanmean(auc_list), np.nanmean(auprc_list)


def subset_indices(groups: Iterable[int], subset: List[str], complement: bool=False) -> Iterable[int]:
    """
    Args
        groups : the group of samples
        subset: 
    """
    if not complement:
        indices = np.arange(len(groups))[np.isin(groups, list(subset))]
    else:
        indices = np.arange(len(groups))[np.logical_not(np.isin(groups, list(subset)))]
    return indices


def _split_train_valid_test(groups, train_keys, valid_keys, test_keys=None):
    """
    groups: length N, the number of samples
    train
    """
    assert isinstance(train_keys, list)
    assert isinstance(valid_keys, list)
    assert test_keys is None or isinstance(test_keys, list)
    index = np.arange(len(groups))
    train_idx = index[np.isin(groups, train_keys)]
    valid_idx = index[np.isin(groups, valid_keys)]
    if test_keys is not None:
        test_idx = index[np.isin(groups, test_keys)]
        return train_idx, valid_idx, test_idx
    else:
        return train_idx, valid_idx

def split_train_valid_test(sample_number: int, val_ratio: float, test_ratio: float, stratify: Iterable=None) -> Tuple[Iterable[int], Iterable[int], Iterable[int]]:
    r"""
    Description
    ------------
    Randomly split train/validation/test data

    Arguments
    ---------
    stratify: split by groups

    Return
    -------
    train_inds : List[int]
    val_inds : List[int]
    test_inds : List[int]
    """
    val_ratio = 0 if val_ratio is None else val_ratio
    test_ratio = 0 if test_ratio is None else test_ratio
    assert val_ratio + test_ratio > 0 and val_ratio + test_ratio < 1, "{},{}".format(val_ratio, test_ratio)
    all_inds = np.arange(sample_number)
    from sklearn.model_selection import train_test_split

    train_val_inds, test_inds = train_test_split(all_inds, test_size=test_ratio, stratify=stratify)
    val_ratio = val_ratio / (1 - test_ratio)
    if stratify is not None:
        stratify = np.asarray(stratify)[train_val_inds]
    train_inds, val_inds = train_test_split(train_val_inds, test_size=val_ratio, stratify=stratify)

    return train_inds, val_inds, test_inds

def split_train_val_test_by_group(groups: List[Any], n_splits: int, val_folds: int, test_folds: int) -> Tuple[List, List, List]:
    from sklearn.model_selection import GroupKFold
    splitter = GroupKFold(n_splits=n_splits)
    train_inds, val_inds, test_inds = list(), list(), list()
    for i, (_, inds) in enumerate(splitter.split(groups, groups=groups)):
        if i < val_folds:
            val_inds.append(inds)
        elif i >= val_folds and i < test_folds + val_folds:
            test_inds.append(inds)
        else:
            train_inds.append(inds)
    train_inds = np.concatenate(train_inds)
    if val_folds > 0:
        val_inds = np.concatenate(val_inds)
    if test_folds:
        test_inds = np.concatenate(test_inds)
    return train_inds, val_inds, test_inds
        

def topk_acc_1d(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    assert labels.ndim == 1 and scores.ndim == 1 and labels.shape[0] == scores.shape[0], "{}".format((labels.shape, scores.shape))
    assert np.issubdtype(labels.dtype, np.bool8) or (np.issubdtype(labels.dtype, np.integer) and labels.max() <= 1 and labels.min() >= 0), "{}".format((labels.dtype, labels.min(), labels.max()))
    labels = labels.astype(np.int8)
    inds = np.where(labels == 1)[0]
    if len(inds) == 0:
        warnings.warn("number of positive sample is 0, return np.nan")
        return np.nan
    else:
        q = np.quantile(scores, q=1 - len(inds) / len(labels))
        return np.where(scores[inds] >= q)[0].shape[0] / np.where(scores >= q)[0].shape[0]

def topk_acc_metrics(labels, scores, ratio:float=1):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    assert labels.ndim == 1 and scores.ndim == 1 and labels.shape[0] == scores.shape[0], "{}".format((labels.shape, scores.shape))
    assert np.issubdtype(labels.dtype, np.bool8) or (np.issubdtype(labels.dtype, np.integer) and labels.max() <= 1 and labels.min() >= 0), "{}".format((labels.dtype, labels.min(), labels.max()))
    labels = labels.astype(np.int8)
    inds = np.where(labels == 1)[0]
    if len(inds) == 0:
        warnings.warn("number of positive sample is 0, return np.nan")
        return np.nan
    else:
        r = ratio * len(inds) / len(labels)
        q = np.quantile(scores, q=1 - r)
        return np.where(scores[inds] >= q)[0].shape[0] / np.where(scores >= q)[0].shape[0]

RegressionMetrics = namedtuple("RegressionMetrics", ["pcc", "scc", "mse"])

def topr_regression_metrics(labels, scores, reference: Literal["true", "pred"], mode: Literal["max", "min"], ratio:float=0.01, regression_metrics: Iterable[Literal["pcc", "scc", "mse"]]=["pcc", "scc", "mse"]) -> RegressionMetrics:
    r"""
    Args 
    ------
    labels : true labels
    scores : predicted scores
    reference : {"true", "pred"}
    mode : {"max", "min"}
    ratio : float (0 < ratio <= 1)

    Return
    -------
    regression_metrics : RegressionMetrics(namedtuple("RegressionMetrics", ["pcc", "scc", "mse"]))
    """
    assert ratio > 0 and ratio <= 1
    assert reference in {"true", "pred"}
    assert mode in {"min", "max"}
    assert labels.ndim == 1 and scores.ndim == 1 and labels.shape[0] == scores.shape[0], "{}".format((labels.shape, scores.shape))
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if ratio < 1:
        if mode == "max":
            ratio = 1 - ratio
        if reference == "true":
            q = np.nanquantile(labels, q=ratio)
            inds = np.where(labels >= q)[0]
        elif reference == "pred":
            q = np.nanquantile(scores, q=ratio)
            inds = np.where(scores >= q)[0]
        labels = labels[inds]
        scores = scores[inds]
    pcc, scc, mse = None, None, None
    if "pcc" in regression_metrics:
        pcc = pearsonr(scores, labels)[0]
    if "scc" in regression_metrics:
        scc = spearmanr(scores, labels)[0]
    if "mse" in regression_metrics:
        mse = metrics.mean_squared_error(scores, labels)
    return RegressionMetrics(pcc=pcc, scc=scc, mse=mse)



# def topk_acc(scores: np.ndarray, labels: np.ndarray, index=None, reduction="mean") -> float:
#     accs = list()
#     if index is None:
#         index = range(1, scores.shape[1])
#     for i in index:
#         inds = np.where(labels == i)[0]
#         if len(inds) == 0:
#             accs.append(np.nan)
#         else:
#             q = np.quantile(scores[:, i], q=1-len(inds) / len(labels))
#             accs.append(sum(scores[inds, i] >= q) / len(inds))
#     if reduction == "mean":
#         return np.mean(accs)
#     elif reduction == "none":
#         return np.asarray(accs)
#     else:
#         raise ValueError("Unknown reduction method: {}".format(reduction))

