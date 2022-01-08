#!/usr/bin/env python3

from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def pr_auc_score(y_true, probas_pred):
    p, r, _ = precision_recall_curve(y_true, probas_pred)
    return auc(r, p)

