# -*- coding: utf-8 -*-
"""复合似然框架：H = ∇²ℓ（sensitivity）、J = Var[∇ℓ]（variability）、Godambe 下界 H⁻¹JH⁻¹。"""

from __future__ import annotations

import numpy as np

from knn_hybrid.model import WeightedDistance, knn_indices


# 参数化
def theta_to_w(theta: np.ndarray) -> np.ndarray:
    """θ ∈ R^{m-1} → w ∈ Δ^{m-1}，最后一个分量固定为参考分量 0。"""
    z = np.append(np.asarray(theta, dtype=float), 0.0)
    e = np.exp(z - z.max())
    return e / e.sum()


def w_to_theta(w: np.ndarray) -> np.ndarray:
    """w（正） → θ，逆映射（差一个常数平移）。"""
    w = np.asarray(w, dtype=float)
    return np.log(np.maximum(w, 1e-12))[:-1] - np.log(np.maximum(w, 1e-12))[-1]


# 复合似然
class CompositeLikelihood:
    """kNN 复合负对数似然 ℓ(θ)。"""

    def __init__(self, X_num, X_cat, y, k: int = 5, mode: str = "soft",
                 sigma_scale: float = 1.0, contribution_rounds: int = 1,
                 eps: float = 1e-12, active: np.ndarray | None = None):
        self.wd = WeightedDistance(np.asarray(X_num, float),
                                   np.asarray(X_cat, int))
        self.y = np.asarray(y, dtype=int)
        self.n = self.wd.n
        self.k = int(k)
        self.mode = mode
        self.eps = eps
        self.m = self.wd.d_num + self.wd.d_cat
        # active：只在给定特征子集上分配权重（其余精确为 0）。
        # 用于把"边界/稀疏导致的方向"分离出去，考察活跃子集上的条件可辨识性。
        self.active = None if active is None else np.asarray(active, dtype=int)
        self.p = (self.m - 1) if self.active is None else (len(self.active) - 1)

        # 贡献度 c：轮次为 1 时只由均匀权重决定，与 w 无关 → 固定，保证 ℓ 光滑
        from knn_hybrid.model import information_gain_contribution
        D0 = self.wd(self.w_uniform(), np.ones(self.n), squared=True)
        idx0 = knn_indices(D0, self.k, ordered=False)
        self.c, _ = information_gain_contribution(self.y, idx0)
        for _ in range(max(1, contribution_rounds) - 1):
            Dw = self.wd(self.w_uniform(), self.c, squared=True)
            idxw = knn_indices(Dw, self.k, ordered=False)
            self.c, _ = information_gain_contribution(self.y, idxw)

        # σ：以均匀权重下第 k 近邻距离的中位数定标，之后固定（不随 w 变 → 光滑）。
        # 用 kNN 半径而不是全体两两距离中位数，k 才真正进入模型（soft 模式下核
        # 覆盖全部样本，若用全体距离定标则 k 完全不起作用，"kNN 复合似然"名不副实）。
        # sigma_scale 的含义：σ = sigma_scale × (典型 kNN 半径)。
        #   < 1 → 比 kNN 半径更局部（趋近硬邻域）；> 1 → 更全局。
        D0 = self.wd(self.w_uniform(), self.c, squared=True)
        dist = np.sqrt(np.maximum(D0, 0.0))
        np.fill_diagonal(dist, np.inf)
        dk = np.sort(dist, axis=1)[:, min(self.k, self.n - 1) - 1]
        self.sigma2 = float(np.median(dk)) ** 2
        self.sigma2 *= sigma_scale ** 2
        self.sigma2 = max(self.sigma2, 1e-9)
        self.d_k = float(np.median(dk))

        self.n_eval = 0

    def w_uniform(self) -> np.ndarray:
        if self.active is None:
            return np.full(self.m, 1.0 / self.m)
        w = np.zeros(self.m)
        w[self.active] = 1.0 / len(self.active)
        return w

    def w_of(self, theta: np.ndarray) -> np.ndarray:
        """θ → w（支持活跃子集参数化）。"""
        if self.active is None:
            return theta_to_w(theta)
        z = np.append(np.asarray(theta, dtype=float), 0.0)
        e = np.exp(z - z.max())
        w = np.zeros(self.m)
        w[self.active] = e / e.sum()
        return w

    def _prob(self, w: np.ndarray) -> np.ndarray:
        """返回 (n, n_cls) 的条件类标概率 p_w(y_i=c | X, y_{-i})。"""
        D2 = self.wd(w, self.c, squared=True)
        n_cls = int(self.y.max()) + 1
        if self.mode == "soft":
            m = np.full(self.n, np.inf)
            dmin = np.min(D2 + np.diag(m), axis=1)          # 每行非对角最小
            K = np.exp(-(D2 - dmin[:, None]) / (2.0 * self.sigma2))
            np.fill_diagonal(K, 0.0)                            # leave-one-out
            q = K / np.maximum(K.sum(axis=1, keepdims=True), 1e-300)
            onehot = np.zeros((self.n, n_cls))
            onehot[np.arange(self.n), self.y] = 1.0
            return q @ onehot
        # hard：top-k 邻域内的距离核加权
        idx = knn_indices(D2, self.k, ordered=False)
        rows = np.arange(self.n)[:, None]
        d = np.sqrt(np.maximum(D2[rows, idx], 0.0))
        p = np.exp(-(d ** 2) / (2.0 * self.sigma2))
        onehot = np.zeros((self.n, n_cls))
        onehot[np.arange(self.n), self.y] = 1.0
        score = (p[:, :, None] * onehot[idx]).sum(axis=1)
        return score / np.maximum(score.sum(axis=1, keepdims=True), 1e-300)

    def nll(self, theta: np.ndarray) -> float:
        self.n_eval += 1
        P = self._prob(self.w_of(theta))
        return float(-np.log(np.maximum(P[np.arange(self.n), self.y], self.eps)).sum())

    def grad(self, theta: np.ndarray, h: float = 1e-2) -> np.ndarray:
        """中心差分梯度 ∇ℓ(θ)。"""
        theta = np.asarray(theta, dtype=float)
        g = np.zeros(self.p)
        for i in range(self.p):
            e = np.zeros(self.p)
            e[i] = h
            g[i] = (self.nll(theta + e) - self.nll(theta - e)) / (2.0 * h)
        return g

    def hessian(self, theta: np.ndarray, h: float = 1e-2) -> np.ndarray:
        """对称化的 O(h²) 中心差分 Hessian ∇²ℓ(θ)。"""
        theta = np.asarray(theta, dtype=float)
        H = np.zeros((self.p, self.p))
        f0 = self.nll(theta)
        for i in range(self.p):
            ei = np.zeros(self.p); ei[i] = h
            for j in range(i, self.p):
                ej = np.zeros(self.p); ej[j] = h
                if i == j:
                    H[i, i] = (self.nll(theta + ei) - 2 * f0
                               + self.nll(theta - ei)) / (h * h)
                else:
                    H[i, j] = H[j, i] = (
                        self.nll(theta + ei + ej) - self.nll(theta + ei - ej)
                        - self.nll(theta - ei + ej) + self.nll(theta - ei - ej)
                    ) / (4.0 * h * h)
        return H

    def directional_curvature(self, theta: np.ndarray, d: np.ndarray,
                              h: float) -> float:
        """沿方向 d 的二阶差分（用于检验硬邻域的不可微性）。"""
        d = np.asarray(d, float)
        d = d / max(np.linalg.norm(d), 1e-12)
        f0 = self.nll(theta)
        return float((self.nll(theta + h * d) - 2 * f0
                      + self.nll(theta - h * d)) / (h * h))


# Godambe / sandwich 分析
def _sym_inv(A: np.ndarray, ridge: float = 1e-9) -> np.ndarray:
    """对称矩阵的（带岭正则）逆。"""
    A = (A + A.T) / 2.0
    ev = np.linalg.eigvalsh(A)
    lam = max(float(np.max(np.abs(ev))) * ridge, 1e-12)
    return np.linalg.inv(A + lam * np.eye(A.shape[0]))


def godambe(H: np.ndarray, J: np.ndarray, clip_neg: bool = True):
    """由 H（sensitivity）与 J（variability）计算 Godambe 信息与各类下界。"""
    Hs = (np.asarray(H, float) + np.asarray(H, float).T) / 2.0
    ev, V = np.linalg.eigh(Hs)
    n_neg = int(np.sum(ev < -1e-9 * max(ev.max(), 1e-12)))
    if clip_neg:
        evc = np.maximum(ev, max(ev.max(), 1e-12) * 1e-6)
        Hs = V @ np.diag(evc) @ V.T
    Hi = _sym_inv(Hs)
    S = Hi @ np.asarray(J, float) @ Hi
    S = (S + S.T) / 2.0
    crb = np.maximum(np.diag(Hi), 0.0)
    sand = np.maximum(np.diag(S), 0.0)
    rho = np.sqrt(sand / np.maximum(crb, 1e-300))
    return {
        "crb": crb,
        "sandwich": sand,
        "rho": rho,
        "rho_trace": float(np.trace(S) / max(np.trace(Hi), 1e-300)),
        "cond_H": float(np.max(np.abs(ev)) / max(np.min(np.abs(ev)), 1e-300)),
        "eigmin_H": float(ev.min()),
        "eigmax_H": float(ev.max()),
        "n_neg": n_neg,
        "H": Hs, "J": J, "Hinv": Hi, "S": S,
    }


def empirical_J(objectives, theta: np.ndarray, h: float = 1e-2) -> np.ndarray:
    """用 B 个独立同分布数据集的经验梯度协方差估计 J = Var[∇ℓ(θ)]。"""
    G = np.array([o.grad(theta, h) for o in objectives])
    return np.cov(G.T, ddof=1)


def empirical_H(objectives, theta: np.ndarray, h: float = 1e-2) -> np.ndarray:
    """多个独立数据集的 Hessian 平均，降低数值噪声。"""
    Hs = [o.hessian(theta, h) for o in objectives]
    return np.mean(Hs, axis=0)


# 数据生成：可控的信息列 / 噪声列
def make_d3_dataset(n=300, d_inf=4, d_noise=2, d_cat=3, n_levels=4,
                    n_clusters=3, sep=1.0, noise_std=0.5, cat_confusion=0.35,
                    seed=0, collinear=False, cat_noise_cols=0, hetero=True,
                    sigma_lo=0.45, sigma_hi=1.30, sigma_noise=3.0,
                    population=None, geometry=None):
    """可控合成数据。"""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_clusters, n)
    if population is None:
        centers = rng.normal(0.0, sep, size=(n_clusters, d_inf))
    else:
        centers = np.asarray(population["centers"], float)

    if hetero and d_inf > 1:
        sig_inf = np.linspace(sigma_lo, sigma_hi, d_inf)
    else:
        sig_inf = np.full(d_inf, float(noise_std))
    X_inf = centers[y] + rng.normal(0.0, 1.0, size=(n, d_inf)) * sig_inf[None, :]
    self_sigma = list(sig_inf)

    cols = [X_inf]
    names = [f"inf_{i}" for i in range(d_inf)]
    if collinear and d_inf >= 1:
        cols.append(X_inf[:, -1:] + rng.normal(0, 1e-6, size=(n, 1)))
        names.append("inf_copy")
        self_sigma.append(self_sigma[-1])
    if d_noise > 0:
        cols.append(rng.normal(0.0, sigma_noise, size=(n, d_noise)))
        names += [f"noise_{i}" for i in range(d_noise)]
        self_sigma += [float(sigma_noise)] * d_noise
    X = np.column_stack(cols)

    cat_cols, cat_names = [], []
    for b in range(d_cat):
        if population is None:
            dominant = rng.integers(0, n_levels, size=n_clusters)
        else:
            dominant = np.asarray(population["cat_dominant"], int)[b]
        vals = np.where(rng.random(n) < cat_confusion,
                        rng.integers(0, n_levels, size=n),
                        dominant[y])
        cat_cols.append(vals)
        cat_names.append(f"cat_{b}")
    for b in range(cat_noise_cols):
        cat_cols.append(rng.integers(0, n_levels, size=n))
        cat_names.append(f"catnoise_{b}")
    X_cat = np.column_stack(cat_cols) if cat_cols else np.zeros((n, 0), dtype=int)

    from knn_hybrid.data import fit_transform_numeric, HybridData
    d = HybridData(fit_transform_numeric(X), X_cat, y,
                   name=f"d3-n{n}-inf{d_inf}-nz{d_noise}",
                   num_names=names, cat_names=cat_names)
    # 理论参考权重：数值列 ∝ 1/σ_j²（高斯判别下的信息量），类属列按混淆率给参考值
    d.sigma = np.asarray(self_sigma, dtype=float)
    return d


def _hypercube_codes(rng, d_inf: int, n_clusters: int) -> np.ndarray:
    '''从 {±1}^d_inf 中按"最远点插入"挑 n_clusters 个码字。'''
    grid = np.array(np.meshgrid(*[[-1.0, 1.0]] * d_inf, indexing="ij"))
    all_codes = grid.reshape(d_inf, -1).T          # (2^d_inf, d_inf)
    if n_clusters > all_codes.shape[0]:
        raise ValueError("n_clusters 超过 2^d_inf 个可用码字")
    start = int(rng.integers(0, all_codes.shape[0]))
    chosen = [start]
    while len(chosen) < n_clusters:
        rest = np.setdiff1d(np.arange(all_codes.shape[0]), np.asarray(chosen))
        # 汉明距离 = (d_inf - 内积)/2
        ip = all_codes[rest] @ all_codes[chosen].T
        ham = (d_inf - ip) / 2.0
        dmin = ham.min(axis=1)
        cand = rest[np.flatnonzero(dmin >= dmin.max())]
        chosen.append(int(rng.choice(cand)))
    return all_codes[np.asarray(chosen)]


def make_d3_population(seed: int = 0, **spec) -> dict:
    '''抽一次"总体"：簇中心 + 类属主值映射。之后所有重复数据集共享它。'''
    rng = np.random.default_rng(seed)
    d_inf = int(spec.get("d_inf", 4))
    d_cat = int(spec.get("d_cat", 3))
    n_clusters = int(spec.get("n_clusters", 3))
    n_levels = int(spec.get("n_levels", 4))
    sep = float(spec.get("sep", 1.0))
    geom = str(spec.get("geometry", "hypercube"))
    if geom == "hypercube":
        codes = _hypercube_codes(rng, d_inf, n_clusters)
        centers = (sep / 2.0) * codes
    else:
        centers = rng.normal(0.0, sep, size=(n_clusters, d_inf))
    return {
        "centers": centers,
        "cat_dominant": rng.integers(0, n_levels, size=(d_cat, n_clusters)),
        "seed": int(seed),
        "geometry": geom,
        "spec": dict(spec),
    }
