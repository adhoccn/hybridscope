# -*- coding: utf-8 -*-
"""去噪顾问 v2：判断这批表格数据该不该做沿样本维的小波去噪。

v1 用的是 lag-1 自相关单阈值(≥0.3 就开去噪)。这与 Paper A 的结论冲突:
lag-1 分不开"真平滑时序"和"按类排序存下来的表" —— 两者 lag-1 都很大,
分开它们的是**自相关曲线的形状**。本模块换成论文里的判据:

  * 结构有无:   acf(1) 中位数 < 0.10 → 样本轴没有结构, 不开
  * 排序键检验: 相邻行同类比例远高于随机 → 行序是存储/排序造成的, 不开
  * 形状比:     τ̂ = acf(8)/acf(1)
                  τ̂ ≥ 0.8   平滑衰减, 真时序 → 开
                  τ̂ ≤ 0.5   快速衰减, 块状伪结构 → 不开
                  0.5<τ̂<0.8 不定带 → 不开, 且如实标注"不定"
  * 有标签时仍以交叉验证为准(直接比准确率), τ̂ 与排序键检验作为交叉校验,
    结论冲突时在 caveat 里说出来, 不静默。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from knn_hybrid.denoise import denoise_numeric, sigma_from_detail
from knn_hybrid.model import WeightedDistance

LAG_LOW = 1
LAG_HIGH = 8
ACF_FLOOR = 0.10        # 低于此值认为样本轴没有结构
TAU_ON = 0.8            # τ̂ ≥ : 开
TAU_OFF = 0.5           # τ̂ ≤ : 不开
BLOCK_RATIO = 2.0       # 相邻同类比例 / 随机期望 > 此值 → 判定为分组块存储


def acf(x, lag: int = 1) -> float:
    """单列自相关（去均值、归一化到 acf(0)=1）。"""
    v = np.asarray(x, float)
    v = v - v.mean()
    denom = float((v * v).sum())
    if denom < 1e-12 or lag >= len(v):
        return 0.0
    return float((v[:-lag] * v[lag:]).sum() / denom)


def acf_profile(X_num: np.ndarray, lags=(1, 2, 4, 8)) -> dict:
    """每列在若干滞后上的自相关，返回 {lag: 各列数组}。"""
    X = np.asarray(X_num, float)
    return {lag: np.array([acf(X[:, j], lag) for j in range(X.shape[1])])
            for lag in lags}


def profile_ratio(X_num: np.ndarray, lag_low: int = LAG_LOW,
                  lag_high: int = LAG_HIGH) -> np.ndarray:
    """τ̂_j = acf(lag_high)/acf(lag_low)，逐列。acf(1) 太小的列记为 NaN。"""
    X = np.asarray(X_num, float)
    out = np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        r1 = acf(X[:, j], lag_low)
        if r1 <= ACF_FLOOR:
            continue
        out[j] = acf(X[:, j], lag_high) / r1
    return out


def block_score(groups) -> dict:
    """排序键检验：行序是不是被某个分组键（班级/专业/标签）排出来的。

    相邻两行同属一组的比例, 与"随机排列下该比例的期望"相比。
    远高于期望说明表是按该键分块存放的, 看上去的平滑是存储顺序造成的。
    """
    codes = pd.factorize(pd.Series(np.asarray(groups)))[0]
    n = len(codes)
    if n < 3:
        return {"same_adj": float("nan"), "expected": float("nan"),
                "ratio": float("nan"), "blocked": False, "n_groups": int(len(set(codes)))}
    same = float((codes[:-1] == codes[1:]).mean())
    _, cnt = np.unique(codes, return_counts=True)
    exp = float(((cnt / n) ** 2).sum())
    blocked = bool(exp < 0.9 and same > BLOCK_RATIO * max(exp, 1e-9))
    return {"same_adj": round(same, 4), "expected": round(exp, 4),
            "ratio": round(same / max(exp, 1e-9), 3), "blocked": blocked,
            "n_groups": int(len(cnt))}


def noise_ratio(X_num: np.ndarray, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    """每列的 σ̂ / std：最细层细节系数的 Donoho 估计占信号标准差的比例。"""
    import pywt
    X = np.asarray(X_num, float)
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        col = X[:, j]
        sd = float(col.std())
        if sd < 1e-12:
            continue
        c = pywt.wavedec(col, wavelet, level=level)
        out[j] = sigma_from_detail(c[-1]) / sd
    return out


def axis_structure(X_num: np.ndarray) -> np.ndarray:
    """每列的 lag-1 自相关（保留接口兼容，内部改用 acf）。"""
    X = np.asarray(X_num, float)
    return np.array([acf(X[:, j], 1) for j in range(X.shape[1])])


def cv_accuracy(X_num, X_cat, y, k: int = 5, folds: int = 5, seed: int = 0) -> float:
    """均匀权重 kNN 的交叉验证准确率。"""
    X_num = np.asarray(X_num, float)
    X_cat = np.asarray(X_cat, int) if X_cat is not None else np.zeros((X_num.shape[0], 0), int)
    y = np.asarray(y, int)
    n = len(y)
    if n < folds * 2:
        return float("nan")

    wd = WeightedDistance(X_num, X_cat)
    m = wd.d_num + wd.d_cat
    D = wd(np.full(m, 1.0 / m), np.ones(n))

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    parts = np.array_split(order, folds)
    n_cls = int(y.max()) + 1
    correct = 0
    for te in parts:
        tr = np.setdiff1d(order, te)
        d = D[np.ix_(te, tr)]
        nn = np.argsort(d, axis=1)[:, : min(k, len(tr))]
        votes = y[tr][nn]
        cnt = np.stack([(votes == c).sum(axis=1) for c in range(n_cls)], axis=1)
        correct += int((cnt.argmax(axis=1) == y[te]).sum())
    return correct / n


def _median(x) -> float:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def recommend(X_num, X_cat=None, y=None, k: int = 5, wavelet: str = "db4",
              level: int = 3, seed: int = 0, margin: float = 0.005,
              groups=None) -> dict:
    """给出去噪开关建议。

    groups: 可选的排序键（班级/专业/批次）。不给时, 有标签用 y, 否则用第一个类别列。
    """
    X_num = np.asarray(X_num, float)
    nr = noise_ratio(X_num, wavelet=wavelet, level=level)
    a1 = axis_structure(X_num)
    tau = profile_ratio(X_num)
    a1_med, tau_med = _median(a1), _median(tau)

    if groups is None:
        if y is not None:
            groups = y
        elif X_cat is not None and np.asarray(X_cat).size:
            groups = np.asarray(X_cat)[:, 0]
    blk = block_score(groups) if groups is not None else None

    info = {
        "noise_ratio": nr,
        "noise_ratio_median": _median(nr),
        "acf1": a1,
        "acf1_median": a1_med,
        "tau": tau,
        "tau_median": tau_med,
        "tau_band": ("no-structure" if not np.isfinite(tau_med) else
                     "smooth" if tau_med >= TAU_ON else
                     "block" if tau_med <= TAU_OFF else "indeterminate"),
        "sort_key": blk,
        "n": int(X_num.shape[0]),
        "d_num": int(X_num.shape[1]),
    }

    if y is None:
        if a1_med < ACF_FLOOR:
            on, rule, reason = False, "no-axis-structure", (
                "无标签：样本轴 lag-1 自相关中位数 %.3f < %.2f，行与行之间没有可借用的结构，"
                "沿样本维去噪只会把真实差异抹平。" % (a1_med, ACF_FLOOR))
        elif blk is not None and blk["blocked"]:
            on, rule, reason = False, "sort-key-blocked", (
                "无标签：排序键检验判定行序由分组键决定（相邻行同组 %.0f%%，随机期望 %.0f%%，"
                "比值 %.1f×），这种平滑是存放顺序造成的，不是数据本身的一维结构。"
                % (blk["same_adj"] * 100, blk["expected"] * 100, blk["ratio"]))
        elif tau_med >= TAU_ON:
            on, rule, reason = True, "smooth-series", (
                "无标签：自相关形状比 τ̂=acf(%d)/acf(%d) 中位数 %.2f ≥ %.1f，"
                "相关缓慢衰减，样本轴构成真正的一维信号，可以去噪。"
                % (LAG_HIGH, LAG_LOW, tau_med, TAU_ON))
        elif tau_med <= TAU_OFF:
            on, rule, reason = False, "block-artifact", (
                "无标签：τ̂ 中位数 %.2f ≤ %.1f，lag-1 高但到 lag-%d 已衰减掉，"
                "是块状伪结构（典型是按某键排序存放），去噪会抹掉块间真实差异。"
                % (tau_med, TAU_OFF, LAG_HIGH))
        else:
            on, rule, reason = False, "indeterminate-band", (
                "无标签：τ̂ 中位数 %.2f 落在不定带 (%.1f, %.1f) 内，数据不足以判定，"
                "按保守原则关闭去噪；如需开启请显式指定 denoise=True 或补充分组/时间信息。"
                % (tau_med, TAU_OFF, TAU_ON))
        info.update({"denoise": on, "rule": rule, "reason": reason})
        return info

    X_d = denoise_numeric(X_num, wavelet=wavelet, level=level)
    acc_raw = cv_accuracy(X_num, X_cat, y, k=k, seed=seed)
    acc_den = cv_accuracy(X_d, X_cat, y, k=k, seed=seed)
    on = bool(acc_den - acc_raw > margin)
    info.update({
        "acc_raw": float(acc_raw),
        "acc_denoised": float(acc_den),
        "delta_acc": float(acc_den - acc_raw),
        "denoise": on,
        "rule": "cv-compare",
        "reason": "交叉验证准确率 %.4f → %.4f（Δ=%+.4f），增益超过 %.3f 才启用去噪。"
                  % (acc_raw, acc_den, acc_den - acc_raw, margin),
    })

    caveats = []
    if on and tau_med <= TAU_OFF and np.isfinite(tau_med):
        caveats.append("CV 说有增益，但 τ̂=%.2f ≤ %.1f（块状伪结构），"
                       "增益可能来自抹平块间差异，建议人工确认。" % (tau_med, TAU_OFF))
    if on and blk is not None and blk["blocked"]:
        caveats.append("排序键检验提示行序由分组键决定（比值 %.1f×），"
                       "去噪增益更可能来自存储顺序而非真实结构。" % blk["ratio"])
    if on and 0.5 < tau_med < 0.8:
        caveats.append("τ̂=%.2f 落在不定带，CV 增益虽为正但依据不强。" % tau_med)
    info["caveat"] = " ".join(caveats)
    return info
