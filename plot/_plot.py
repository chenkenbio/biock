#!/usr/bin/env python3

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
new_rc_params = {'text.usetex': False, 'svg.fonttype': 'none' }
plt.rcParams.update(new_rc_params)
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

def hide_spines(ax: Axes, left=True, right=False, top=False, bottom=True):
    ax.spines["left"].set_visible(left)
    ax.spines["right"].set_visible(right)
    ax.spines["top"].set_visible(top)
    ax.spines["bottom"].set_visible(bottom)

def boxplot_with_scatter(x, size, ax=None, max_sample=None, scatter_color: List[str]=None, **kwargs):
    if ax is None:
        ax = plt.subplot()
    
    if "sym" not in kwargs:
        kwargs["sym"] = ''
    bb = ax.boxplot(x=x, **kwargs)
    # bb2 = ax.boxplot(x=x2, **kwargs)
    
    if not hasattr(x[0], "__iter__"):
        x = [x]
    
    for i, ind in enumerate(ax.get_xticks()):
        ar = np.asarray(x[i])
        if max_sample is not None and len(ar) > max_sample:
            ar = np.random.permutation(ar)[:max_sample]
        left, right = bb["boxes"][i].get_data()[0][:2]
        xs = np.random.randn(len(ar))
        xmin, xmax = xs.min(), xs.max()
        xs =  (2 * (xs - xmin) / (xmax - xmin) - 1) * (right - left) / 2 + ind
        if type(scatter_color) is str:
            c = scatter_color
        elif hasattr(scatter_color, "__iter__"):
            c = scatter_color[i]
        else:
            c = None
        scatter = ax.scatter(xs, ar, s=size, c=c)

