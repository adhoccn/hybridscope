# -*- coding: utf-8 -*-
"""顶 混合属性数据去噪 （§2.1，denoise.py） kNN 聚类模型 （§2.2，model.py）"""

from __future__ import annotations

import numpy as np

from .denoise import denoise_categorical, denoise_numeric
from .model import HybridKNNModel, WeightedDistance, knn_indices, knn_predict, mutual_knn_clustering
from .metrics import all_metrics
from .pso import PSO


class KNNHybridClustering:
    def __init__(self, k: int = 5,
                 denoise: bool = True, wavelet: str = "db4", level: int = 3,
                 max_shift: int = 20, threshold_rule: str = "bayes",
                 cat_smooth: bool = False, cat_smooth_k: int = 5,
                 denoise_sort_by: str = "none", denoise_rescale: bool = False,
                 pso_particles: int = 30, pso_iters: int = 50,
                 encoding: str = "weight", use_pso: bool = True,
                 contribution_rounds: int = 2, loss: str = "mse",
                 low_memory: bool = False, seed: int = 0):
        self.k = k
        self.denoise = denoise
        self.wavelet = wavelet
        self.level = level
        self.max_shift = max_shift
        self.threshold_rule = threshold_rule
        self.cat_smooth = cat_smooth
        self.cat_smooth_k = cat_smooth_k
        self.denoise_sort_by = denoise_sort_by
        self.denoise_rescale = denoise_rescale
        self.pso_particles = pso_particles
        self.pso_iters = pso_iters
        self.encoding = encoding
        self.use_pso = use_pso
        self.contribution_rounds = contribution_rounds
        self.loss = loss
        self.low_memory = low_memory
        self.seed = seed

        self.w_ = None
        self.c_ = None
        self.history_ = None
        self.time_denoise_ = 0.0
        self.time_pso_ = 0.0

    def fit(self, X_num, X_cat, y=None):
        import time

        X_num = np.asarray(X_num, dtype=float)
        X_cat = np.asarray(X_cat, dtype=int) if X_cat is not None else np.zeros((X_num.shape[0], 0), int)
        n = X_num.shape[0]
        self.y_ = None if y is None else np.asarray(y, dtype=int)

# 去噪 ---- #
        t0 = time.perf_counter()
        if self.denoise and X_num.shape[1] > 0:
            X_num_d = denoise_numeric(X_num, wavelet=self.wavelet, level=self.level,
                                      max_shift=self.max_shift,
                                      threshold_rule=self.threshold_rule,
                                      sort_by=self.denoise_sort_by,
                                      rescale=self.denoise_rescale)
        else:
            X_num_d = X_num.copy()
        if self.cat_smooth and X_cat.shape[1] > 0:
            X_cat_d = denoise_categorical(X_cat, X_num_d, k=self.cat_smooth_k)
        else:
            X_cat_d = X_cat.copy()
        self.time_denoise_ = time.perf_counter() - t0
        self.X_num_ = X_num_d
        self.X_cat_ = X_cat_d

# 建模型---- #
        if self.y_ is None:
            D0 = WeightedDistance(X_num_d, X_cat_d)(np.ones(X_num_d.shape[1] + X_cat_d.shape[1]),
                                                    np.ones(n))
            self.labels_ = mutual_knn_clustering(D0, self.k)
            return self

        model = HybridKNNModel(X_num_d, X_cat_d, self.y_, k=self.k,
                               contribution_rounds=self.contribution_rounds,
                               loss=self.loss, low_memory=self.low_memory)
        self.model_ = model

# PSO  ---- #
        t0 = time.perf_counter()
        if self.use_pso:
            pso = PSO(dim=model.dim, n_particles=self.pso_particles,
                      iters=self.pso_iters, seed=self.seed)
            if self.encoding == "weight":
                w, f, hist = pso.optimize(model.evaluate, encoding="weight")
            else:
                def fit_neighbor(S):
                    D = model.wd(model.w_uniform, np.ones(n))
                    rows = np.arange(n)
                    dsub = D[:, S]
                    n_cls = int(self.y_.max()) + 1
                    sigma = np.median(dsub) + 1e-9
                    p = np.exp(-(dsub ** 2) / (2 * sigma ** 2))
                    onehot = np.zeros((n, n_cls))
                    onehot[rows, self.y_] = 1.0
                    score = p @ onehot[S]
                    prob = score / (score.sum(axis=1, keepdims=True) + 1e-12)
                    return float(np.mean(np.sum((prob - onehot) ** 2, axis=1)))

                S, f, hist = pso.optimize(fit_neighbor, encoding="neighbor", n_items=n)
                # 均匀权重 + 额外近邻集合
                w = model.w_uniform.copy()
                self.neighbor_set_ = S
            self.w_ = w
            self.history_ = hist
        else:
            self.w_ = model.w_uniform.copy()
        self.time_pso_ = time.perf_counter() - t0

        loss, info = model.evaluate(self.w_, return_detail=True)
        self.c_ = info["c"]
        self.labels_ = info["y_pred"]
        self.train_loss_ = loss
        self.train_acc_ = info["acc"]
        return self

    def predict(self, X_num_new, X_cat_new) -> np.ndarray:
        """加权 kNN 预测，新样本贡献度取常数 1"""
        wd_new = WeightedDistance(
            np.vstack([self.X_num_, np.asarray(X_num_new, dtype=float)]),
            np.vstack([self.X_cat_, np.asarray(X_cat_new, dtype=int)])
            if self.X_cat_ is not None and self.X_cat_.shape[1] else
            np.vstack([self.X_cat_, np.asarray(X_cat_new, dtype=int)]),
        )
        n_tr = self.X_num_.shape[0]
        c_full = np.concatenate([self.c_ if self.c_ is not None else np.ones(n_tr),
                                 np.ones(np.asarray(X_num_new).shape[0])])
        D = wd_new(self.w_, c_full)
        Dsub = D[n_tr:, :n_tr]
        idx = np.argpartition(Dsub, min(self.k, n_tr) - 1, axis=1)[:, : self.k]
        d = np.take_along_axis(Dsub, idx, axis=1)
        sigma = np.median(d) + 1e-9
        p = np.exp(-(d ** 2) / (2 * sigma ** 2))
        n_cls = int(self.y_.max()) + 1
        onehot = np.zeros((n_tr, n_cls))
        onehot[np.arange(n_tr), self.y_] = 1.0
        score = np.einsum("ik,kc->ic", p, onehot[idx].sum(axis=1) / 1.0) \
            if False else (p[:, :, None] * onehot[idx]).sum(axis=1)
        return score.argmax(axis=1)

    def score(self, y_true=None) -> dict:
        y_true = self.y_ if y_true is None else np.asarray(y_true, dtype=int)
        return all_metrics(y_true, self.labels_)
