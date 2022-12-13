#!/usr/bin/env python3

import argparse, os, sys, warnings, time, json, gzip
import numpy as np

from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    f1_score, matthews_corrcoef, 
    precision_score, recall_score, accuracy_score, 
    roc_curve, precision_recall_curve
)

from collections import OrderedDict, defaultdict
import biock
from biock import auto_open, count_items
from biock.ml_tools import topk_acc_1d, pr_auc_score
from scipy.stats import pearsonr, spearmanr



def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', help="label\tscore\t...", nargs='+', required=True)
    p.add_argument('-t', "--task", choices=("C", "R", "classification", "regression"), help="classification(C)/regression(R) metrics", default="classification")
    p.add_argument("-m", "--metrics", nargs='+', choices=("AUC", "AP", "F1", "topK-ACC", "PCC", "SCC", "MSE"))
    p.add_argument('-c', '--cutoff', default=0.5, type=float, help="cutoff used in F1/MCC calculation")
    p.add_argument('-r', "--reverse", action='store_true', help="reverse score")
    p.add_argument("-l", "--label-column", default=0, help="column ID of label (0-start)", type=int)
    p.add_argument("-s", "--score-column", default=1, help="column ID of score (0-start)", type=int)
    p.add_argument("--skip-labels", type=str, nargs='+')
    p.add_argument("--skiprows", default=0, type=int)
    p.add_argument("--comment", default="#", type=str)
    p.add_argument('-d', "--delimiter", default='\t')
    p.add_argument("--allow-nan", action="store_true")
    p.add_argument("--single-line", action="store_true")
    # p.add_argument('-roc', action='store_true', help="save fpr(X)/tpr(Y) for plotting ROC curve")
    # p.add_argument('-pr', action='store_true', help="save recall(X)/precision(Y) for plotting ROC curve")
    p.add_argument("--f1-options", type=str, help="kwargs in f1_score")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()

    label, score = list(), list()

    skip_labels = dict()
    if args.skip_labels is not None:
        for k in args.skip_labels:
            skip_labels[k] = 0


    for fn in args.input:
        with auto_open(fn) as infile:
            for nr, line in enumerate(infile):
                if nr < args.skiprows or line.startswith(args.comment):
                    continue
                fields = line.strip("\n").split(args.delimiter)
                l, s = fields[args.label_column], fields[args.score_column]
                if l in skip_labels:
                    skip_labels[l] += 1
                label.append(float(l))
                score.append(float(s))
    label = np.asarray(label).astype(int)
    score = np.asarray(score)
    nan_scores = None
    if args.allow_nan:
        keep = ~np.isnan(score)
        nan_scores = count_items(label[~keep], fraction=True)
        label = label[keep]
        score = score[keep]
    if args.reverse:
        score = -score

    results = OrderedDict()
    classification_metrics = ["AUC", "AP", "AUPR", "topK-ACC", "F1", "MCC", "precision", "recall", "ACC"]
    if args.task == "classification" or args.task == "C":
        assert len(np.unique(label)) == 2, "{}".format(np.unique(label, return_counts=True))
        if args.metrics is None:
            args.metrics = classification_metrics
        else:
            raise NotImplementedError
        pred = (score > args.cutoff).astype(int)
        for m in args.metrics:
            if m == "AUC":
                results[m] = roc_auc_score(label, score)
            elif m == "AP":
                results[m] = average_precision_score(label, score)
            elif m == "AUPR":
                results[m] = pr_auc_score(label, score)
            elif m == "topK-ACC":
                results[m] = topk_acc_1d(label, score)
            elif m == "F1":
                results["{}(c={:g})".format(m, args.cutoff)] = f1_score(label, pred)
            elif m == "MCC":
                results["{}(c={:g})".format(m, args.cutoff)] = matthews_corrcoef(label, pred)
            elif m == "precision":
                results["{}(c={:g})".format(m, args.cutoff)] = precision_score(label, pred)
            elif m == "recall":
                results["{}(c={:g})".format(m, args.cutoff)] = recall_score(label, pred)
            elif m == "ACC":
                results["{}(c={:g})".format(m, args.cutoff)] = accuracy_score(label, pred)
            else:
                raise ValueError("unknown metric {}".format(m))

        print(biock.get_run_info(sys.argv, args))
        print("#labels skipped: {}".format(skip_labels))
        if nan_scores is not None:
            print("#nan scores skipped: {}".format(nan_scores))
        print("#label count: {}".format(biock.count_items(label, fraction=True)).replace(',', ' ,').replace('(', '( ').replace(')', ' )'))
        if args.single_line:
            print('#' + '\t'.join([x for x in results.keys()]))
            print('\t'.join(["{:.4f}".format(x) for x in results.values()]))
        else:
            for m, s in results.items():
                print("-{}:\t{:.5f}".format(m, s))
    else:
        raise NotImplementedError("do not support {} task".format(args.task))
