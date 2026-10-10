# -*- coding: utf-8 -*-
"""对比算法实现（复现论文 §3 的实验比较）。"""

from __future__ import annotations

import numpy as np


# 文献[3]：熵离散化 + 混合距离 k-modes
def _split_entropy_gain(values: np.ndarray, cut: float, labels: np.ndarray):
    left = labels[values <= cut]
    right = labels[values > cut]
    if left.size == 0 or right.size == 0:
        return -np.inf

    def ent(idx):
        if idx.size == 0:
            return 0.0
        p = np.bincount(idx) / idx.size
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    n = values.size
    return -((left.size / n) * ent(left) + (right.size / n) * ent(right))


def entropy_discretize(col: np.ndarray, labels: np.ndarray, max_bins: int = 4) -> np.ndarray:
    """基于类标熵增益的数值属性离散化（MDLP 的简化版），返回整数编码。"""
    col = np.asarray(col, dtype=float)
    n = col.size
    codes = np.zeros(n, dtype=int)
    parts = [(np.arange(n), 0)]
    for _ in range(max_bins):
        parts.sort(key=lambda p: p[0].size, reverse=True)
        idx, base = parts.pop(0)
        vals = col[idx]
        labs = labels[idx]
        if idx.size < 6 or len(np.unique(vals)) < 2:
            parts.append((idx, base))
            break
        cand = np.unique(vals)
        cand = (cand[:-1] + cand[1:]) / 2.0
        if cand.size > 32:  # 采样候选切点，控制开销
            cand = cand[np.linspace(0, cand.size - 1, 32).astype(int)]
        gains = [_split_entropy_gain(vals, c, labs) for c in cand]
        best = int(np.argmax(gains))
        if not np.isfinite(gains[best]):
            parts.append((idx, base))
            break
        cut = cand[best]
        l = idx[vals <= cut]
        r = idx[vals > cut]
        parts.append((l, base * 2))
        parts.append((r, base * 2 + 1))
    for idx, base in parts:
        codes[idx] = base
    return codes


class EntropyKPrototypes:
    """文献[3]：基于熵的混合属性聚类算法。"""

    def __init__(self, n_clusters: int, max_iter: int = 50, seed: int = 0):
        self.K = n_clusters
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

    def fit_predict(self, X_num: np.ndarray, X_cat: np.ndarray, y=None) -> np.ndarray:
        n = X_num.shape[0]
        labels_for_disc = y if y is not None else np.zeros(n, dtype=int)
        A = np.column_stack(
            [entropy_discretize(X_num[:, j], labels_for_disc) for j in range(X_num.shape[1])]
        ) if X_num.shape[1] else np.zeros((n, 0), dtype=int)
        X = np.column_stack([A, X_cat]) if X_cat.shape[1] else A

        # 随机选择 K 个初始聚类中心
        init = self.rng.choice(n, self.K, replace=False)
        centers = X[init].copy()
        assign = np.zeros(n, dtype=int)
        for _ in range(self.max_iter):
            # 二次元（混合）距离：逐属性不匹配计数
            D = np.array([(X != centers[k]).sum(axis=1) for k in range(self.K)]).T
            new = D.argmin(axis=1)
            if np.array_equal(new, assign):
                break
            assign = new
            for k in range(self.K):
                mem = X[assign == k]
                if mem.size == 0:
                    continue
                for b in range(X.shape[1]):
                    vals, cnts = np.unique(mem[:, b], return_counts=True)
                    centers[k, b] = vals[np.argmax(cnts)]
        return assign


# 文献[4]：黏菌算法 + 身份搜索优化的模糊 C 均值（SMA-IS-FCM）
def _fcm_objective(X_num, X_cat, centers_num, centers_cat, m=2.0, lam=1.0):
    """混合属性 FCM：给定中心计算隶属度与目标函数值。"""
    n, K = X_num.shape[0], centers_num.shape[0]
    d2 = ((X_num[:, None, :] - centers_num[None, :, :]) ** 2).sum(axis=2) if X_num.shape[1] else \
        np.zeros((n, K))
    if X_cat.shape[1]:
        hamm = np.zeros((n, K))
        for k in range(K):
            hamm[:, k] = (X_cat != centers_cat[k][None, :]).sum(axis=1)
        d2 = d2 + lam * hamm
    d2 = np.maximum(d2, 1e-12)
    power = -2.0 / (m - 1.0)
    u = d2 ** power
    u = u / u.sum(axis=1, keepdims=True)
    J = float(np.sum((u ** m) * d2))
    return J, u


class SMAFuzzyCMeans:
    """文献[4]：基于混合身份搜索黏菌优化的模糊 C-均值聚类。"""

    def __init__(self, n_clusters: int, n_pop: int = 20, iters: int = 30,
                 m: float = 2.0, lam: float = 1.0, z: float = 0.03, seed: int = 0):
        self.K = n_clusters
        self.P = n_pop
        self.T = iters
        self.m = m
        self.lam = lam
        self.z = z
        self.rng = np.random.default_rng(seed)

    def fit_predict(self, X_num: np.ndarray, X_cat: np.ndarray) -> np.ndarray:
        n, d = X_num.shape
        K = self.K
        lo, hi = X_num.min(axis=0), X_num.max(axis=0)
        dim = K * d

        # 分类中心先用 k-modes 思路初始化（众数）
        centers_cat = np.zeros((K, X_cat.shape[1]), dtype=int) if X_cat.shape[1] else np.zeros((K, 0), int)
        init = self.rng.choice(n, K, replace=False)
        if X_cat.shape[1]:
            centers_cat = X_cat[init].copy()

        def objective(vec):
            cn = vec.reshape(K, d)
            J, u = _fcm_objective(X_num, X_cat, cn, centers_cat, self.m, self.lam)
            return J, u

        X = self.rng.uniform(0, 1, (self.P, dim)) * (hi - lo)[None, :].repeat(K, axis=1) \
            if False else self.rng.uniform(lo.min(), hi.max(), (self.P, dim))
        fitness = np.array([objective(x)[0] for x in X])
        DF = float(fitness.min())
        best_idx = int(np.argmin(fitness))

        for t in range(self.T):
            order = np.argsort(fitness)
            bF, wF = float(fitness[order[0]]), float(fitness[order[-1]])
            tt = (t + 1) / self.T
            a = float(np.arctanh(np.clip(1.0 - tt, 1e-6, 0.999999)))
            b = 1.0 - t / self.T

            # 黏菌权重 W
            W = np.zeros((self.P, dim))
            for r, i in enumerate(order):
                if r < self.P / 2:
                    W[i, :] = 1 + self.rng.random() * np.log10((bF - fitness[i]) / (wF - bF + 1e-12) + 1)
                else:
                    W[i, :] = 1 - self.rng.random() * np.log10((bF - fitness[i]) / (wF - bF + 1e-12) + 1)

            for i in range(self.P):
                p = np.tanh(abs(fitness[i] - DF))
                vb = self.rng.uniform(-a, a, dim)
                vc = self.rng.uniform(-b, b, dim)
                if self.rng.random() < self.z:  # 随机重启
                    X[i] = self.rng.uniform(lo.min(), hi.max(), dim)
                else:
                    A, B = self.rng.integers(0, self.P, 2)
                    rnd = self.rng.random(dim)
                    mask = rnd < p
                    X[i][mask] = X[best_idx][mask] + vb[mask] * (W[i][mask] * X[A][mask] - X[B][mask])
                    X[i][~mask] = vc[~mask] * X[i][~mask]
                X[i] = np.clip(X[i], lo.min(), hi.max())
                fitness[i] = objective(X[i])[0]
                if fitness[i] < DF:
                    DF = float(fitness[i])
                    best_idx = i

            # 身份搜索（IS）：对最优个体做局部扰动，接受更优解
            cand = X[best_idx] + self.rng.normal(0, 0.05, dim) * (hi.max() - lo.min())
            f_cand = objective(cand)[0]
            if f_cand < fitness[best_idx]:
                X[best_idx] = cand
                fitness[best_idx] = f_cand
                DF = float(f_cand)

            # 依据最优个体的隶属度更新分类中心（众数）
            J, u = objective(X[best_idx])
            if X_cat.shape[1]:
                for k in range(K):
                    w = u[:, k]
                    for bb in range(X_cat.shape[1]):
                        vals = np.unique(X_cat[:, bb])
                        scores = [w[X_cat[:, bb] == v].sum() for v in vals]
                        centers_cat[k, bb] = vals[int(np.argmax(scores))]

        J, u = objective(X[best_idx])
        return u.argmax(axis=1)


# 消融基线：经典小波变换去噪 + 均匀权重 kNN（不含贡献度与 PSO 权重）
def wavelet_knn_baseline(X_num, X_cat, y, k=5, wavelet="db4", level=3):
    from .denoise import denoise_numeric
    from .model import HybridKNNModel

    Xd = denoise_numeric(X_num, wavelet=wavelet, level=level, max_shift=0,
                         threshold_rule="universal")
    model = HybridKNNModel(Xd, X_cat, y, k=k, contribution_rounds=1)
    loss, info = model.evaluate(model.w_uniform, return_detail=True)
    return info["y_pred"], info["acc"], loss
