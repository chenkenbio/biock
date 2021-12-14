#!/usr/bin/env python3
"""
Author: Ken Chen (chenkenbio@gmail.com)
Date: <<date>>
"""

from ..biock import nt_onehot_dict
import numpy as np


def onehot_dna_rna(seq):
    return np.array([nt_onehot_dict[n] for n in seq])
# import logging, warnings, json, gzip, pickle
# from collections import defaultdict, OrderedDict
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu
# from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score



