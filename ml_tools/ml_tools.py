#!/usr/bin/env python3

from typing import Iterable, Tuple
import sklearn.metrics as metrics # auc, precision_recall_curve, roc_auc_score
import numpy as np

def roc_auc_score(y_true, probas_pred, robust: bool=False) -> float:
    if robust and len(np.unique(y_true)) == 1:
        res = 0.5
    else:
        res = metrics.roc_auc_score(y_true, probas_pred)
    return res


def pr_auc_score(y_true, probas_pred, robust: bool=False):
    if robust and len(np.unique(y_true)) == 1:
        res = 0
    else:
        p, r, _ = metrics.precision_recall_curve(y_true, probas_pred)
        res = metrics.auc(r, p)
    return res


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
    y_true, probas_pred = np.asarray(y_true), np.asarray(probas_pred)
    assert np.array_equal(y_true.shape, probas_pred.shape), "expected equal size array, but {} and {} were found".format(y_true.shape, probas_pred.shape)
    assert len(y_true.shape) == 2, "2d array is expected, while encountered {}".format(y_true.shape)
    auc_list, auprc_list = list(), list()
    for i in range(y_true.shape[0]):
        auc_list.append(roc_auc_score(y_true[i], probas_pred[i], robust=robust))
        auprc_list.append(pr_auc_score(y_true[i], probas_pred[i], robust=True))
    return np.mean(auc_list), np.mean(auprc_list)


