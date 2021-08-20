#!/usr/bin/env python3

import argparse, os, sys, warnings, time, json, gzip
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, matthews_corrcoef, f1_score, average_precision_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

from functools import partial

def cal_aupr(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    aupr = auc(recall, precision)
    return (aupr, precision, recall)


def label_count(labels):
    """ labels should be list,np.array """
    categories, counts = np.unique(labels, return_counts=True)
    ratio = (counts / counts.sum()).round(3)
    return list(zip(categories, counts, ratio))

def average_FDR_FOR(y_true, y_prob):
    N = len(y_true)
    y_true, y_prob = np.array(y_true).astype(int), np.array(y_prob) 
    cutoffs = np.linspace(min(y_prob) + 1E-6, max(y_prob) - 1E-6, 1000)
    fdr_sum = 0
    for_sum = 0
    for c in cutoffs:
        pred = (y_prob >= c).astype(int)
        FDR = np.logical_and(y_true == 0, pred == 1).sum() / np.sum(pred)
        FOR = np.logical_and(y_true == 1, pred == 0).sum() / (N - np.sum(pred))
        fdr_sum += FDR
        for_sum += FOR
    return fdr_sum / N, for_sum / N

def min_FDR_FOR(y_true, y_prob):
    N = len(y_true)
    y_true, y_prob = np.array(y_true).astype(int), np.array(y_prob) 
    cutoffs = np.linspace(min(y_prob) + 1E-6, max(y_prob) - 1E-6, 1000)
    min_fdr = 10
    min_for = 10
    for c in cutoffs:
        pred = (y_prob >= c).astype(int)
        FDR = np.logical_and(y_true == 0, pred == 1).sum() / np.sum(pred)
        FOR = np.logical_and(y_true == 1, pred == 0).sum() / (N - np.sum(pred))
        if FDR < min_fdr:
            min_fdr = FDR
        if FOR < min_for:
            min_for = FOR
    return min_fdr, min_for

def max_f1(y_true, y_prob):
    max_f1 = -1
    best_cutoff = None
    for cutoff in np.linspace(min(y_prob) + 1E-6, max(y_prob) - 1E-6, 100):
        pred = (y_prob >= cutoff).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > max_f1:
            best_cutoff = cutoff
            max_f1 = f1
    return max_f1, best_cutoff
            
def max_mcc(y_true, y_prob):
    max_mcc = -1
    best_cutoff = None
    for cutoff in np.linspace(min(y_prob) + 1E-6, max(y_prob) - 1E-6, 100):
        pred = (y_prob >= cutoff).astype(int)
        mcc = matthews_corrcoef(y_true, pred)
        if mcc > max_mcc:
            best_cutoff = cutoff
            max_mcc = mcc
    return max_mcc, best_cutoff
 


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('file')
    p.add_argument('-f', '--f1-cutoff', default=None, type=float)
    p.add_argument('-r', action='store_true')
    p.add_argument('--roc', action='store_true')
    p.add_argument('--aupr', action='store_true')
    p.add_argument('--pos', type=int, default=1)
    p.add_argument('--neg', type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()

    label, score = list(), list()

    if args.file.endswith("gz"):
        custom_open = partial(gzip.open, mode='rt')
    else:
        custom_open = partial(open, mode='rt')
    with custom_open(args.file) as infile:
        for l in infile:
            if l.startswith('#') or len(l.strip()) == 0:
                continue
            l, s = l.strip().split()[0:2]
            label.append(float(l))
            score.append(float(s))
    label = np.array(label)
    score = np.array(score)
    if len(np.unique(label)) > 2:
        label, score = score, label
    if args.r:
        score = -score

    label_score = zip(list(score), list(label))
    label_score = sorted(label_score, key=lambda l:l[0], reverse=True)

    
    n_pos = int(sum(label))
    top10 = sum([a[1] for a in label_score[0:10]])     / min(n_pos, 10)
    top100 = sum([a[1] for a in label_score[0:100]])   / min(n_pos, 100)
    top200 = sum([a[1] for a in label_score[0:200]])   / min(n_pos, 200)
    top500 = sum([a[1] for a in label_score[0:500]])   / min(n_pos, 500)
    top1000 = sum([a[1] for a in label_score[0:1000]]) / min(n_pos, 1000)

    AUC = roc_auc_score(label, score)
    AP = average_precision_score(label, score)
    AUPR = cal_aupr(label, score)[0]
    print("# {}".format(args.file))
    print("# label count: {}".format(label_count(label)).replace(',', ' ,').replace('(', '( ').replace(')', ' )'))
    print("- AUC:  {:.5f}".format(AUC))
    print("- AUPR: {:.5f}".format(AUPR))
    print("- AP:   {:.5f}".format(AP))
    if args.f1_cutoff is None:
        F1, cutoff = max_f1(label, score)
        args.f1_cutoff = cutoff
    pred = (score > args.f1_cutoff).astype(int)
    N = len(pred)
    F1 = f1_score(label, pred)
    print("\n- F1:   {:.5f} (cutoff = {:.5f})".format(F1, args.f1_cutoff).replace(')', ' )'))
    FDR = np.logical_and(label == 0, pred == 1).sum() / np.sum(pred)
    FOR = np.logical_and(label == 1, pred == 0).sum() / (N - np.sum(pred))
    print(" - FDR:  {:.5f}".format(FDR))
    print(" - FOR:  {:.5f}\n".format(FOR))

    mcc, mcc_cutoff = max_mcc(label, score)
    print("- MCC: {:.5f} (cutoff = {:.5f})\n".format(mcc, mcc_cutoff).replace(')', ' )'))

    print("- Top10   ACC: {:.1f}".format(top10))
    print("- Top100  ACC: {:.2f}".format(top100))
    print("- Top200  ACC: {:.2f}".format(top200))
    print("- Top500  ACC: {:.3f}".format(top500))
    print("- Top1000 ACC: {:.3f}".format(top1000))

    #mean_FDR, mean_FOR = average_FDR_FOR(label, score)
    #print("- Average FDR: {:.5f}".format(mean_FDR))
    #print("- Average FOR: {:.5f}".format(mean_FOR))

