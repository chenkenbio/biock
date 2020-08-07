#!/usr/bin/env python3

import argparse, os, sys, time
#import warnings, json, gzip
import numpy as np

class EarlyStopping(object):
    def __init__(self, patience=7, verbose=False, delta=0, score_fun="AUC"):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.scores = dict()
        self.use_auc = False
        self.use_aupr = False
        self.score_fun = score_fun.lower()
        if self.score_fun not in {'auc', 'aupr', 'auprc', 'loss', 'auc+aupr'}:
            raise ValueError("Unexpected score_fun: '{}'".format(self.score_fun))
        self.n_epoch = 0

    def __call__(self, **kwargs):
        self.n_epoch += 1
        for k in kwargs:
            if k.lower() == "auc":
                if "AUC" not in self.scores:
                    self.scores['AUC'] = list()
                self.scores["AUC"].append(kwargs[k])
                epoch_auc = kwargs[k]
            elif k.lower() == "aupr" or k.lower() == "auprc":
                if "AUPR" not in self.scores:
                    self.scores['AUPR'] = list()
                self.scores["AUPR"].append(kwargs[k])
                epoch_aupr = kwargs[k]
            elif k.lower() == "loss":
                if "loss" not in self.scores:
                    self.scores['loss'] = list()
                self.scores['loss'].append(kwargs['loss'])
                epoch_loss = kwargs[k]
        if self.score_fun == "auc":
            epoch_score = epoch_auc
        elif self.score_fun == "aupr":
            epoch_score = epoch_aupr
        elif score_fun == "loss":
            epoch_score = -epoch_loss
        elif self.score_fun == "auc+aupr":
            epoch_score = (epoch_auc + epoch_aupr) / 2
        else:
            raise ValueError("Unexpected score_fun: '{}'".format(self.score_fun))

        if self.best_score is None:
            self.best_score = epoch_score
            self.best_epoch = self.n_epoch
        elif epoch_score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print("- EarlyStop log: {:.5f} < {:.5f}, epoch {:d}".format(epoch_score, self.best_score, self.n_epoch))
            if self.counter >= self.patience:
                self.early_stop = True
        elif epoch_score > self.best_score:
            self.best_score = epoch_score
            self.best_epoch = self.n_epoch
            self.counter = 0



def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--seed', type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    #np.random.seed(args.seed)

    x = np.random.rand(100)
    x = np.random.rand(100)
    print(x)

    es = EarlyStopping(verbose=True)
    for AUC in x:
        es(AUC=AUC)
        if es.early_stop:
            print(es.best_epoch, es.best_score)
            break
