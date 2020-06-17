#!/usr/bin/env python3

import argparse, os, sys, warnings, time, json, gzip
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, matthews_corrcoef, f1_score, average_precision_score
from scipy.stats import pearsonr, spearmanr


def cal_aupr(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    aupr = auc(recall, precision)
    return (aupr, precision, recall)


def label_count(labels):
    """ labels should be list,np.array """
    categories, counts = np.unique(labels, return_counts=True)
    ratio = (counts / counts.sum()).round(3)
    return list(zip(categories, counts, ratio))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('file')
    p.add_argument('--roc', action='store_true')
    p.add_argument('--aupr', action='store_true')
    p.add_argument('--pos', type=int, default=1)
    p.add_argument('--neg', type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()

    label, score = list(), list()
    with open(args.file) as infile:
        for l in infile:
            if l.startswith('#') or len(l.strip()) == 0:
                continue
            l, s = l.strip().split()[0:2]
            label.append(int(l))
            score.append(float(s))
    label = np.array(label)
    score = np.array(score)
    AUC = roc_auc_score(label, score)
    AP = average_precision_score(label, score)
    AUPR = cal_aupr(label, score)[0]
    print("# {}".format(args.file))
    print("# label count: {}".format(label_count(label)))
    print("- AUC:  {:.5f}".format(AUC))
    print("- AUPR: {:.5f}".format(AUPR))
    print("- AP:   {:.5f}".format(AP))
