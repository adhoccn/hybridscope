# -*- coding: utf-8 -*-
"""沿样本维的小波阈值去噪：自适应窗口噪声估计 + BayesShrink 阈值 + 循环平移。"""

from __future__ import annotations

import numpy as np
import pywt

K_DONOHO = 0.6745  # 高斯噪声下 median(|d|)/sigma 的理论常数


# 高频系数噪声标准差估计
def sigma_from_detail(detail: np.ndarray) -> float:
    """由第 j 层细节系数估计噪声标准差（Donoho 中值估计）。"""
    if detail.size == 0:
        return 0.0
    return float(np.median(np.abs(detail)) / K_DONOHO)


# 滑动窗口自适应噪声方差估计
def adaptive_window_sigma(detail: np.ndarray, win_len: int = 32, step: int = 8,
                          rounds: int = 3, tau_scale: float = 1.0):
    """滑动窗口自适应噪声标准差估计。"""
    d = np.asarray(detail, dtype=float)
    n = d.size
    if n < 8:
        return sigma_from_detail(d), 1.0, []

    sigma0 = sigma_from_detail(d)
    if sigma0 <= 1e-12:  # 无噪声
        return 0.0, 1.0, []

    win_len = int(np.clip(win_len, 4, n))
    step = int(np.clip(step, 1, win_len))
    hist = []
    rho = 1.0
    sigma = sigma0

    for _ in range(rounds):
        # 窗口滑动，逐窗口求标准差（式3）
        stds = []
        for s in range(0, n - win_len + 1, step):
            stds.append(d[s:s + win_len].std())
        stds = np.asarray(stds)
        if stds.size == 0:
            stds = np.array([d.std()])

        # 阈值 tau 与比值 rho（式4）
        tau = tau_scale * sigma0
        rho = float(np.mean(stds < tau))

        # 依据 rho 重设窗口长度与滑动步长
        # rho 越小（起伏越大）→ 缩短窗口以跟踪局部；rho 越大 → 放大窗口降低方差
        win_len = int(np.clip(round(win_len * (0.6 + 0.8 * (1.0 - rho))), 4, n))
        step = int(np.clip(round(win_len / 2), 1, win_len))
        hist.append((win_len, step, rho))

    # 用最终窗口重估并排序，按 rho 分位确定噪声标准差
    stds = []
    for s in range(0, n - win_len + 1, max(1, step)):
        stds.append(d[s:s + win_len].std())
    stds = np.sort(np.asarray(stds) if stds else np.array([d.std()]))
    q = float(np.clip(rho, 0.05, 0.95))
    sigma = float(np.quantile(stds, q)) / K_DONOHO * K_DONOHO  # 已是标准差量级
    return float(sigma), rho, hist


#BayesShrink 最佳阈值
def bayesshrink_threshold(coeffs: np.ndarray, sigma: float) -> float:
    """式(5)：lambda = sigma^2 / sigma_X，sigma_X^2 = max(var(c) - sigma^2, 0)。"""
    if sigma <= 1e-12:
        return 0.0
    var_c = float(np.var(coeffs))
    sigma_x = np.sqrt(max(var_c - sigma ** 2, 1e-12))
    return float(sigma ** 2 / sigma_x)


def universal_threshold(coeffs: np.ndarray, sigma: float) -> float:
    """对照（经典小波）：通用阈值 sigma * sqrt(2 ln N)。"""
    n = max(coeffs.size, 2)
    return float(sigma * np.sqrt(2.0 * np.log(n)))


# 循环平移（cycle spinning）去噪
def _shift(x: np.ndarray, k: int) -> np.ndarray:
    return np.roll(x, k)


def wavelet_denoise_1d(x: np.ndarray, wavelet: str = "db4", level: int = 3,
                       max_shift: int = 20, mode: str = "periodization",
                       threshold_rule: str = "bayes", sigma: float | None = None,
                       soft: bool = True) -> np.ndarray:
    """对一维信号做循环平移小波阈值去噪 threshold_rule : 'bayes'（自适应窗口 + BayesShrink）或 'universal'（经典小波）"""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 8:
        return x.copy()
    max_shift = int(max(0, min(max_shift, n - 1)))
    shifts = range(0, max_shift + 1)
    acc = np.zeros(n)
    cnt = 0
    for k in shifts:
        xs = _shift(x, k)
        coeffs = pywt.wavedec(xs, wavelet, level=level, mode=mode)
        if len(coeffs) < 2:
            continue
        detail = coeffs[-1]
        if sigma is not None:
            s = sigma
        elif threshold_rule == "bayes":
            s, _, _ = adaptive_window_sigma(detail)
        else:
            s = sigma_from_detail(detail)
        if s <= 1e-12:
            rec = xs
        else:
            newc = [coeffs[0]]
            for c in coeffs[1:]:
                thr = bayesshrink_threshold(c, s) if threshold_rule == "bayes" \
                    else universal_threshold(c, s)
                newc.append(pywt.threshold(c, thr, mode="soft" if soft else "hard"))
            rec = pywt.waverec(newc, wavelet, mode=mode)
            if rec.size > n:
                rec = rec[:n]
            elif rec.size < n:
                rec = np.pad(rec, (0, n - rec.size))
        acc += _shift(rec, -k)
        cnt += 1
    if cnt == 0:
        return x.copy()
    return acc / cnt


def _pca_order(X: np.ndarray) -> np.ndarray:
    
    Xc = np.asarray(X, dtype=float) - X.mean(axis=0)
    if Xc.shape[1] == 1:
        score = Xc[:, 0]
    else:
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        score = Xc @ vt[0]
    return np.argsort(score)


def denoise_numeric(X: np.ndarray, wavelet="db4", level=3, max_shift=20,
                    threshold_rule="bayes", soft=True, sort_by="none",
                    rescale=False) -> np.ndarray:
    """对数值属性矩阵逐列（逐属性）去噪。"""
    X = np.asarray(X, dtype=float)
    order = np.arange(X.shape[0]) if sort_by == "none" else _pca_order(X)
    Xs = X[order]
    out = np.empty_like(Xs)
    for j in range(Xs.shape[1]):
        out[:, j] = wavelet_denoise_1d(
            Xs[:, j], wavelet=wavelet, level=level, max_shift=max_shift,
            threshold_rule=threshold_rule, soft=soft,
        )
    inv = np.argsort(order)
    out = out[inv]
    if rescale:
        sd = out.std(axis=0)
        sd[sd < 1e-12] = 1.0
        out = (out - out.mean(axis=0)) / sd
    return out


def denoise_categorical(X_cat: np.ndarray, X_num: np.ndarray, k: int = 5) -> np.ndarray:
    """k 近邻众数投票平滑。
    """
    if X_cat.shape[1] == 0 or X_num.shape[1] == 0:
        return X_cat.copy()
    n = X_cat.shape[0]
    D = ((X_num[:, None, :] - X_num[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(D, np.inf)
    k = min(k, n - 1)
    idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    out = X_cat.copy()
    for i in range(n):
        neigh = X_cat[idx[i]]
        for b in range(X_cat.shape[1]):
            vals, cnts = np.unique(neigh[:, b], return_counts=True)
            out[i, b] = vals[np.argmax(cnts)]
    return out


def snr_db(clean: np.ndarray, observed: np.ndarray) -> float:
    """信噪比 SNR = 10 log10( P_signal / P_noise )。"""
    noise = np.asarray(observed, dtype=float) - np.asarray(clean, dtype=float)
    ps = float(np.sum(np.asarray(clean, dtype=float) ** 2))
    pn = float(np.sum(noise ** 2))
    if pn <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(ps / pn)
