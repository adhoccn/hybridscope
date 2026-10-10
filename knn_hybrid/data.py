# -*- coding: utf-8 -*-
"""混合属性数据的加载 / 合成 / 预处理。"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field

import numpy as np


@dataclass
class HybridData:
    """混合属性数据容器。"""

    X_num: np.ndarray
    X_cat: np.ndarray
    y: np.ndarray
    name: str = "dataset"
    num_names: list = field(default_factory=list)
    cat_names: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.X_num.shape[0]

    @property
    def d_num(self) -> int:
        return self.X_num.shape[1]

    @property
    def d_cat(self) -> int:
        return self.X_cat.shape[1]

    @property
    def dim(self) -> int:
        """特征权重向量的维度（数值属性 + 分类属性）。"""
        return self.d_num + self.d_cat

    def subset(self, idx) -> "HybridData":
        idx = np.asarray(idx)
        return HybridData(
            self.X_num[idx],
            self.X_cat[idx],
            self.y[idx],
            self.name,
            self.num_names,
            self.cat_names,
        )


# 预处理
def fit_transform_numeric(X: np.ndarray) -> np.ndarray:
    """数值属性 z-score 标准化（论文中用于消除混合属性尺度不一致）。"""
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


def encode_categorical(X) -> np.ndarray:
    """分类属性整数编码，返回 int 数组与每列的取值个数。"""
    import pandas as pd

    df = pd.DataFrame(X)
    cols = []
    n_cat_levels = []
    for c in df.columns:
        codes, _ = pd.factorize(df[c])
        cols.append(codes.astype(int))
        n_cat_levels.append(int(codes.max()) + 1)
    return np.column_stack(cols) if cols else np.zeros((len(df), 0), dtype=int)


def n_cat_levels(X_cat: np.ndarray) -> np.ndarray:
    """每列分类属性的取值个数。"""
    if X_cat.shape[1] == 0:
        return np.zeros(0, dtype=int)
    return (X_cat.max(axis=0) + 1).astype(int)


# 合成混合属性数据集（可控维度 / 规模 / 噪声）
def make_hybrid_dataset(
    n_samples=300,
    d_num=6,
    d_cat=4,
    n_clusters=3,
    n_levels=4,
    sep=1.2,
    noise_std=0.35,
    cat_confusion=0.30,
    seed=0,
) -> HybridData:
    """生成混合属性数据集。"""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_clusters, n_samples)
    centers = rng.normal(0.0, sep, size=(n_clusters, d_num))

    X_num = centers[y] + rng.normal(0.0, noise_std, size=(n_samples, d_num))

    X_cat = np.zeros((n_samples, d_cat), dtype=int)
    for b in range(d_cat):
        dominant = rng.integers(0, n_levels, size=n_clusters)
        vals = np.where(
            rng.random(n_samples) < cat_confusion,
            rng.integers(0, n_levels, size=n_samples),
            dominant[y],
        )
        X_cat[:, b] = vals

    return HybridData(
        fit_transform_numeric(X_num),
        X_cat,
        y,
        name=f"synthetic-n{n_samples}-d{d_num + d_cat}",
        num_names=[f"num_{i}" for i in range(d_num)],
        cat_names=[f"cat_{i}" for i in range(d_cat)],
    )


def make_smooth_hybrid_dataset(
    n_samples=300, d_num=6, d_cat=4, n_clusters=3, n_levels=4,
    snr_db=6.0, cat_confusion=0.25, seed=0,
) -> HybridData:
    """混合属性数据集。"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi, n_samples)
    amp = rng.uniform(0.6, 1.4, size=(d_num, 1))
    freq = rng.uniform(0.5, 2.5, size=(d_num, 1))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(d_num, 1))
    signal = (amp * np.sin(freq * t[None, :] + phase)).T  # (n, d_num)

    y = np.minimum((t / (2 * np.pi) * n_clusters).astype(int), n_clusters - 1)

    sig_p = signal.var(axis=0)
    noise = rng.normal(0, 1, signal.shape) * np.sqrt(sig_p / (10 ** (snr_db / 10)))
    X_num = signal + noise

    X_cat = np.zeros((n_samples, d_cat), dtype=int)
    for b in range(d_cat):
        dominant = rng.integers(0, n_levels, size=n_clusters)
        X_cat[:, b] = np.where(rng.random(n_samples) < cat_confusion,
                               rng.integers(0, n_levels, size=n_samples),
                               dominant[y])
    # 用干净信号的统计量统一标准化，保证 SNR 可精确计算
    mu, sd = signal.mean(axis=0), signal.std(axis=0)
    sd[sd < 1e-12] = 1.0
    clean_s = (signal - mu) / sd
    noisy_s = (X_num - mu) / sd
    common = dict(num_names=[f"num_{i}" for i in range(d_num)],
                  cat_names=[f"cat_{i}" for i in range(d_cat)])
    return (
        HybridData(clean_s, X_cat, y, name=f"smooth-clean-n{n_samples}-d{d_num + d_cat}", **common),
        HybridData(noisy_s, X_cat, y, name=f"smooth-noisy-{snr_db}dB", **common),
    )


def add_noise(data: HybridData, snr_db: float = 5.0, seed: int = 0) -> HybridData:
    """按给定信噪比给数值属性叠加高斯白噪声"""
    rng = np.random.default_rng(seed)
    sig_p = data.X_num.var(axis=0)
    noise_var = sig_p / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, 1.0, data.X_num.shape) * np.sqrt(noise_var)
    noisy = HybridData(
        data.X_num + noise,
        data.X_cat,
        data.y,
        data.name + f"-noisy{snr_db}dB",
        data.num_names,
        data.cat_names,
    )
    return noisy


# UCI / OpenML 数据集
def load_openml(name: str, target: str, max_samples: int = 600, seed: int = 0):
    """尝试从 OpenML 拉取混合属性数据集；失败返回 None。"""
    import pandas as pd
    from sklearn.datasets import fetch_openml

    old = socket.getdefaulttimeout()
    socket.setdefaulttimeout(12)
    try:
        ds = fetch_openml(name=name, version=1, as_frame=True, parser="auto")
    except Exception:
        return None
    finally:
        socket.setdefaulttimeout(old)

    df = ds.frame.dropna()
    if target not in df.columns:
        return None
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=seed).reset_index(drop=True)

    y = df[target].astype("category").cat.codes.to_numpy()
    X = df.drop(columns=[target])
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    X_num = X[num_cols].to_numpy(dtype=float) if num_cols else np.zeros((len(X), 0))
    X_cat = encode_categorical(X[cat_cols]) if cat_cols else np.zeros((len(X), 0), dtype=int)
    return HybridData(
        fit_transform_numeric(X_num),
        X_cat,
        y,
        name=f"openml:{name}",
        num_names=num_cols,
        cat_names=cat_cols,
    )


def get_benchmark_datasets(seed: int = 0):
    """返回测试数据集
    """
    specs = [
        ("cmc", "ContraceptiveMethodUsed", 400, dict(n_samples=300, d_num=4, d_cat=5, n_clusters=3, seed=seed + 1)),
        ("credit-g", "class", 400, dict(n_samples=300, d_num=7, d_cat=13, n_clusters=2, seed=seed + 2)),
        ("heart-statlog", "class", 300, dict(n_samples=300, d_num=6, d_cat=7, n_clusters=2, seed=seed + 3)),
    ]
    out = []
    for name, target, cap, synth in specs:
        d = load_openml(name, target, cap, seed)
        if d is None:
            d = make_hybrid_dataset(**synth)
            d.name = f"synthetic[{name}]-n{synth['n_samples']}-d{synth['d_num'] + synth['d_cat']}"
        out.append(d)
    return out
