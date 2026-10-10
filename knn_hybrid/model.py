# -*- coding: utf-8 -*-
"""混合属性 kNN：信息增益贡献度 → 特征加权距离 → kNN 预测 → 损失。"""

from __future__ import annotations

import numpy as np


# 样本贡献度（信息增益）
def _entropy(counts: np.ndarray) -> float:
    p = counts / max(counts.sum(), 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def information_gain_contribution(y: np.ndarray, nbr_idx: np.ndarray,
                                  lo: float = 0.5, hi: float = 1.5,
                                  include_self: bool = True):
    """计算每个样本的贡献度（全向量化）。"""
    y = np.asarray(y, dtype=int)
    n = y.size
    n_cls = int(y.max()) + 1
    onehot = np.zeros((n, n_cls))
    onehot[np.arange(n), y] = 1.0

    cnt = onehot[nbr_idx].sum(axis=1)          # (n, n_cls)
    if include_self:
        cnt = cnt + onehot

    def _ent(counts):
        s = counts.sum(axis=1, keepdims=True)
        p = counts / np.maximum(s, 1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(p > 0, -p * np.log(p), 0.0)
        return term.sum(axis=1)

    h_global = float(_ent(onehot.sum(axis=0, keepdims=True))[0])
    ig = h_global - _ent(cnt)

    ig = ig - ig.min()
    rng = ig.max() - ig.min()
    if rng > 1e-12:
        c = lo + (ig / rng) * (hi - lo)
    else:
        c = np.full(n, (lo + hi) / 2.0)
    return c.astype(float), float(ig.mean())


# 加权距离
class WeightedDistance:
    """计算融入贡献度的特征权重欧氏距离。

    low_memory=True 时不预先物化类别相等指示张量 E（n×n×d_cat），改成逐列
    累加。结果数值等价，但峰值内存从 O(n²·d_cat) 降到 O(n²)：大表上 E 往往
    比距离矩阵本身还大，是第一个爆内存的地方。
    """

    def __init__(self, X_num: np.ndarray, X_cat: np.ndarray,
                 low_memory: bool = False):
        self.X_num = np.asarray(X_num, dtype=float)
        self.X_cat = np.asarray(X_cat, dtype=int)
        self.d_num = self.X_num.shape[1]
        self.d_cat = self.X_cat.shape[1]
        self.n = self.X_num.shape[0]
        self.low_memory = bool(low_memory)
        self.E = None
        if not self.low_memory and self.d_cat > 0:
            # 分类属性相等指示张量 E[:, :, b] = 1[a_ib == a_jb]
            self.E = np.zeros((self.n, self.n, self.d_cat), dtype=float)
            for b in range(self.d_cat):
                col = self.X_cat[:, b]
                self.E[:, :, b] = (col[:, None] == col[None, :]).astype(float)

    def __call__(self, w: np.ndarray, c: np.ndarray, squared: bool = False) -> np.ndarray:
        """返回距离矩阵 D (n, n)。w 维度 = d_num + d_cat，c 维度 = n。"""
        w = np.asarray(w, dtype=float)
        w_num, w_cat = w[: self.d_num], w[self.d_num:]
        c = np.asarray(c, dtype=float)

        if self.d_num > 0:
            U = self.X_num * c[:, None] * np.sqrt(np.maximum(w_num, 0.0))[None, :]
            G = U @ U.T
            r = np.diag(G)
            d2 = r[:, None] + r[None, :]
            d2 -= 2.0 * G
        else:
            d2 = np.zeros((self.n, self.n))

        if self.d_cat > 0:
            # delta_ij = Σ_b w_b * 1[a_ib != a_jb]
            if self.E is not None:
                eq = np.tensordot(self.E, w_cat, axes=([2], [0]))  # Σ_b w_b E_ijb
            else:                                  # 逐列累加，省掉 n²·d_cat 的张量
                eq = np.zeros((self.n, self.n), dtype=float)
                for b in range(self.d_cat):
                    col = self.X_cat[:, b]
                    eq += float(w_cat[b]) * (col[:, None] == col[None, :])
            delta = np.maximum(w_cat.sum() - eq, 0.0)
            scale = (c[:, None] + c[None, :])
            scale *= 0.5
            scale *= scale
            d2 += scale * delta

        np.maximum(d2, 0.0, out=d2)
        np.fill_diagonal(d2, 0.0)
        return d2 if squared else np.sqrt(d2)


# k 近邻与预测
def knn_indices(D: np.ndarray, k: int, exclude_self: bool = True,
                ordered: bool = True) -> np.ndarray:
    """返回每个样本的 k 个最近邻索引 (n, k)。

    排除自身时不复制整个 D（旧做法要 D.copy() 再把对角线填 inf）：多取一个
    近邻再剔掉自身即可。省掉的是每次评估都要走的 n² 拷贝，PSO 跑几百次时
    这部分内存带宽很可观。
    """
    n = D.shape[0]
    if n <= 1:
        return np.zeros((n, 0), dtype=int)
    k = int(max(1, min(k, n - 1)))
    rows = np.arange(n)[:, None]
    if exclude_self:
        kk = min(k + 1, n)
        idx = np.argpartition(D, kk - 1, axis=1)[:, :kk]
        mask = idx != rows                       # 自身距离为 0，必在这 kk 个里
        idx = idx[mask].reshape(n, kk - 1)[:, :k]
    else:
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    if not ordered:
        return idx
    order = np.argsort(D[rows, idx], axis=1)
    return idx[rows, order]


def knn_predict(D: np.ndarray, y: np.ndarray, k: int = 5, exclude_self: bool = True,
                squared: bool = False):
    """距离加权 kNN 预测，返回 (y_pred, prob, loss_mse)。"""
    y = np.asarray(y, dtype=int)
    n = y.size
    n_cls = int(y.max()) + 1
    idx = knn_indices(D, k, exclude_self, ordered=False)
    rows = np.arange(n)[:, None]
    d = D[rows, idx]
    sigma2 = np.median(d) + 1e-9 if squared else (np.median(d) + 1e-9) ** 2
    p = np.exp(-d / (2.0 * sigma2)) if squared else np.exp(-(d ** 2) / (2.0 * sigma2))  # 式(9)

    onehot = np.zeros((n, n_cls))
    onehot[np.arange(n), y] = 1.0
    score = (p[:, :, None] * onehot[idx]).sum(axis=1)  # 式(10)
    score_sum = score.sum(axis=1, keepdims=True) + 1e-12
    prob = score / score_sum
    y_pred = prob.argmax(axis=1)
    loss = float(np.mean(np.sum((prob - onehot) ** 2, axis=1)))  # 式(11)(12)
    return y_pred, prob, loss


def mutual_knn_clustering(D: np.ndarray, k: int = 5) -> np.ndarray:
    """无监督聚类：互 k 近邻图连通分量（无类标时使用）。"""
    n = D.shape[0]
    idx = knn_indices(D, k, exclude_self=True)
    adj = np.zeros((n, n), dtype=bool)
    adj[np.repeat(np.arange(n), idx.shape[1]), idx.ravel()] = True
    adj = adj & adj.T  # mutual kNN
    labels = -np.ones(n, dtype=int)
    c = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        stack, comp = [i], []
        labels[i] = c
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.where(adj[u])[0]:
                if labels[v] < 0:
                    labels[v] = c
                    stack.append(int(v))
        c += 1
    return labels


# 完整聚类模型的一次前向评估
class HybridKNNModel:
    """给定权重 w，完成"贡献度 → 加权距离 → kNN 预测 → 损失"""

    def __init__(self, X_num, X_cat, y, k: int = 5, contribution_rounds: int = 1,
                 loss: str = "mse", low_memory: bool = False):
        self.wd = WeightedDistance(X_num, X_cat, low_memory=low_memory)
        self.y = np.asarray(y, dtype=int)
        self.k = k
        self.rounds = contribution_rounds
        self.loss = loss
        self.w_uniform = np.ones(self.wd.d_num + self.wd.d_cat)
        self.w_uniform /= self.w_uniform.sum()
        # 均匀权重 + 单位贡献度下的距离矩阵与 w 无关，每次评估都重算它是纯浪费。
        # PSO 跑几百次，这一步缓存直接省掉 rounds/(rounds+1) 里的一整次 n² 计算。
        self._D_uniform = None

    @property
    def dim(self) -> int:
        return self.wd.d_num + self.wd.d_cat

    def evaluate(self, w: np.ndarray, return_detail: bool = False):
        """返回 loss；return_detail=True 时附加 acc / y_pred / c / D。"""
        w = np.asarray(w, dtype=float)
        s = w.sum()
        w = w / s if s > 1e-12 else self.w_uniform.copy()

        # 均匀权重下的近邻 → 贡献度
        if self._D_uniform is None:
            self._D_uniform = self.wd(self.w_uniform, np.ones(self.wd.n), squared=True)
        D = self._D_uniform
        idx = knn_indices(D, self.k, ordered=False)
        c, _ = information_gain_contribution(self.y, idx)

        # 贡献度 + 权重的距离刷新近邻与贡献度
        for _ in range(max(1, self.rounds) - 1):
            D = self.wd(w, c, squared=True)
            idx = knn_indices(D, self.k, ordered=False)
            c, _ = information_gain_contribution(self.y, idx)

        D = self.wd(w, c, squared=True)
        y_pred, prob, loss_mse = knn_predict(D, self.y, self.k, squared=True)
        if self.loss == "err":
            loss = float(np.mean(y_pred != self.y))
        else:
            loss = loss_mse
        if return_detail:
            return loss, {
                "acc": float(np.mean(y_pred == self.y)),
                "y_pred": y_pred,
                "c": c,
                "D": np.sqrt(D),
                "prob": prob,
            }
        return loss
