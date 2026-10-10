# -*- coding: utf-8 -*-
"""聚类准确率"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def clustering_accuracy(y_true, y_pred) -> float:
    """ACC匈牙利算法"""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = y_true.size
    K = max(int(y_true.max()), int(y_pred.max())) + 1
    C = np.zeros((K, K))
    for i in range(n):
        C[y_pred[i], y_true[i]] += 1
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].sum() / n)


def nmi(y_true, y_pred) -> float:
    return float(normalized_mutual_info_score(y_true, y_pred))


def ari(y_true, y_pred) -> float:
    return float(adjusted_rand_score(y_true, y_pred))


def all_metrics(y_true, y_pred) -> dict:
    return {
        "ACC": clustering_accuracy(y_true, y_pred),
        "NMI": nmi(y_true, y_pred),
        "ARI": ari(y_true, y_pred),
    }


@contextmanager
def timer():
    t0 = time.perf_counter()
    holder = {}
    try:
        yield holder
    finally:
        holder["t"] = time.perf_counter() - t0
