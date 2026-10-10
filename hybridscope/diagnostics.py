# -*- coding: utf-8 -*-
"""权重可估计性诊断：把 Godambe 下界换算成每个特征的置信半径与红黄绿灯。"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from knn_hybrid.estimability import CompositeLikelihood, theta_to_w, w_to_theta

EIG_FLOOR_REL = 1e-4
N_CAP = 10 ** 6
COND_LIMIT = 1e6
# 数值 Hessian 用有限差分（h=1e-2）算，噪声会带来极小的负特征值；
# 只有相对最大特征值超过 1e-3 的负曲率才算"真的不在极大值点上"。
NEG_REL = 1e-3


def _jacobian_w(theta: np.ndarray):
    """softmax 值与雅可比 ∂w/∂θ（最后一维为参考分量）。"""
    w = theta_to_w(theta)
    m = w.size
    Jw = np.diag(w) - np.outer(w, w)
    return w, Jw[:, : m - 1]


def _refine(cl: CompositeLikelihood, theta: np.ndarray) -> np.ndarray:
    """以给定点为初值做一次局部极小化，让 H 落在真正的驻点上。"""
    try:
        r = minimize(cl.nll, np.asarray(theta, float), method="Powell",
                     options={"maxiter": 2000, "xtol": 1e-4, "ftol": 1e-6})
        if np.isfinite(r.fun) and r.fun <= cl.nll(theta) + 1e-9:
            return np.asarray(r.x, float)
    except Exception:
        pass
    return np.asarray(theta, float)


def _per_sample_grads(cl: CompositeLikelihood, theta: np.ndarray, h: float):
    """逐样本梯度 g_i = ∇ℓ_i(θ)，用于 J = n·Var[g_i]。"""
    n, p = cl.n, cl.p
    idx = np.arange(n)
    G = np.zeros((n, p))
    for j in range(p):
        e = np.zeros(p)
        e[j] = h
        lp = -np.log(np.maximum(cl._prob(cl.w_of(theta + e))[idx, cl.y], 1e-300))
        lm = -np.log(np.maximum(cl._prob(cl.w_of(theta - e))[idx, cl.y], 1e-300))
        G[:, j] = (lp - lm) / (2.0 * h)
    return G


def _regularized_inverse(H: np.ndarray):
    """把 H 的最小特征值抬到 λmax·EIG_FLOOR_REL 以上再求逆。"""
    Hs = (H + H.T) / 2.0
    ev, V = np.linalg.eigh(Hs)
    lam_max = max(float(np.max(np.abs(ev))), 1e-12)
    floor = lam_max * EIG_FLOOR_REL
    n_neg = int(np.sum(ev < -NEG_REL * lam_max))
    evc = np.maximum(ev, floor)
    Hi = V @ np.diag(1.0 / evc) @ V.T
    return Hi, n_neg, float(np.max(np.abs(ev)) / max(float(np.min(np.abs(evc))), 1e-300))


def weight_uncertainty(X_num, X_cat, y, theta, k: int = 5, h: float = 1e-2,
                       mode: str = "soft", refine: bool = True,
                       sigma_scale: float = 1.0, n_boot: int = 0,
                       seed: int = 0):
    """在 θ* 处估计 H 与 J，给出 w 空间的标准误。"""
    X_num = np.asarray(X_num, float)
    X_cat = np.asarray(X_cat, int) if X_cat is not None else np.zeros((X_num.shape[0], 0), int)
    y = np.asarray(y, int)

    cl = CompositeLikelihood(X_num, X_cat, y, k=k, mode=mode,
                             sigma_scale=sigma_scale)
    theta_star = _refine(cl, theta) if refine else np.asarray(theta, float)

    H = cl.hessian(theta_star, h)
    G = _per_sample_grads(cl, theta_star, h)
    # 只有两个特征时 θ 是一维的，np.cov 会退化成标量；atleast_2d 补回矩阵形状。
    # 不做这一步，任何 m=2 的分析（统一口径层正是两个 canonical 字段）都拿不到标准误。
    J = cl.n * np.atleast_2d(np.cov(G.T, ddof=1))

    Hi, n_neg, cond = _regularized_inverse(H)
    S = Hi @ J @ Hi
    S = (S + S.T) / 2.0
    crb = np.maximum(np.diag(Hi), 0.0)
    nll = float(cl.nll(theta_star))
    nll_ps = nll / max(cl.n, 1)

    w, A = _jacobian_w(theta_star)
    S_w = A @ S @ A.T
    se_w = np.sqrt(np.maximum(np.diag(S_w), 0.0))

    # 梯度 bootstrap：只重抽样本维度的得分贡献（θ 固定），检验 J 的稳定性。
    # 比"重抽后重新拟合"便宜得多，能反映 sandwich 对个别样本的依赖程度。
    se_w_boot = None
    if int(n_boot) > 0 and G.shape[0] > 3:
        rng = np.random.default_rng(seed)
        n_s = G.shape[0]
        acc = np.zeros((int(n_boot), w.size))
        for b in range(int(n_boot)):
            idx = rng.integers(0, n_s, n_s)
            Gb = G[idx]
            Jb = cl.n * np.atleast_2d(np.cov(Gb.T, ddof=1))
            Sb = Hi @ Jb @ Hi
            Sb = (Sb + Sb.T) / 2.0
            acc[b] = np.sqrt(np.maximum(np.diag(A @ Sb @ A.T), 0.0))
        se_w_boot = acc.mean(axis=0)

    unstable = bool(n_neg > 0 or cond > COND_LIMIT)
    return {
        "theta_star": theta_star,
        "w": w,
        "se_w": se_w,
        "se_w_boot": se_w_boot,
        "crb_theta": crb,
        "sandwich_theta": np.maximum(np.diag(S), 0.0),
        "rho": np.sqrt(np.maximum(np.diag(S), 0.0) / np.maximum(crb, 1e-300)),
        "rho_trace": float(np.trace(S) / max(np.trace(Hi), 1e-300)),
        "cond_H": cond,
        "n_neg": n_neg,
        "unstable": unstable,
        "identifiable": bool(not unstable and np.all(np.isfinite(se_w))),
        "separable": bool(nll_ps < 0.01),
        "nll": nll,
        "nll_per_sample": nll_ps,
        "H": H, "J": J, "Hinv": Hi, "S": S,
    }


def traffic_light(w: np.ndarray, se: np.ndarray, boundary: float = 1e-3):
    """与均匀权重 1/m 比较的 z 值；权重落在单纯形边界时单独成档。"""
    w = np.asarray(w, float)
    se = np.asarray(se, float)
    z = (w - 1.0 / w.size) / np.maximum(se, 1e-12)
    az = np.abs(z)
    level = np.where(az >= 2.0, 2, np.where(az >= 1.0, 1, 0))
    on_boundary = w < boundary
    return np.where(on_boundary, 0.0, z), np.where(on_boundary, -1, level)


def two_sided_p(z: np.ndarray) -> np.ndarray:
    """z 值转双尾 p（正态近似）。z 用于与等权 1/m 比较，双尾。"""
    z = np.asarray(z, float)
    return 2.0 * norm.sf(np.abs(z))


def bh_fdr(p: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray, int]:
    """Benjamini-Hochberg：返回 (q 值, 是否显著, 显著个数)。

    m 个权重是同时检验的，逐个看 z > 2 会把假阳性率放大到约 1-(1-α)^m；
    BH 控制的是错误发现率（显著结果里假阳性的期望比例），比 Bonferroni
    保守、比不校正严谨，是多权重场景的标准做法。
    """
    p = np.asarray(p, float)
    m = p.size
    if m == 0:
        return p, np.zeros(0, bool), 0
    order = np.argsort(p)
    ps = p[order]
    q = ps * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]   # 单调化
    q = np.clip(q, 0.0, 1.0)
    qv = np.empty(m, float)
    qv[order] = q
    sig = qv <= float(alpha)
    return qv, sig, int(sig.sum())


LEVEL_NAME = {2: "green", 1: "amber", 0: "red", -1: "boundary"}
LEVEL_ZH = {2: "可信", 1: "勉强", 0: "不可辨识", -1: "未启用(边界)"}


def sample_size_plan(n: int, se: np.ndarray, target: float) -> np.ndarray:
    """CRB ∝ 1/N：把标准误压到 target 所需样本量，超过 N_CAP 记为不可达。"""
    need = np.ceil(n * (np.asarray(se, float) / max(float(target), 1e-12)) ** 2)
    return np.minimum(need.astype(np.int64), N_CAP)


def diagnose(X_num, X_cat, y, w, k: int = 5, seed: int = 0,
             feature_names=None, target_se: float = 0.02,
             fdr_alpha: float = 0.05, n_boot: int = 0, **kw) -> dict:
    """一次调用返回全部诊断量，含 BH-FDR 多重校正后的结论档位。"""
    w = np.asarray(w, float)
    res = weight_uncertainty(X_num, X_cat, y, w_to_theta(w), k=k,
                             n_boot=n_boot, seed=seed, **kw)
    z, level = traffic_light(res["w"], res["se_w"])
    n = int(len(y))
    need = sample_size_plan(n, res["se_w"], target_se)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(w.size)]

    # 多重校正：m 个权重同时检验，逐个看 |z|>=2 会放大假阳性。
    p = two_sided_p(z)
    q, sig, n_sig = bh_fdr(p, fdr_alpha)
    # 校正后档位：显著 + |z|>=2 → 可信；显著 + |z|>=1 → 勉强；其余不可辨识。
    lvl_fdr = np.where(sig & (np.abs(z) >= 2.0), 2,
                       np.where(sig & (np.abs(z) >= 1.0), 1, 0))
    lvl_fdr = np.where(level < 0, -1, lvl_fdr)          # 边界权重不参与
    lvl_fdr = np.where(np.isfinite(p), lvl_fdr, 0)

    identifiable = bool(res["identifiable"])
    if not identifiable:
        # SE 不可信 → p/q 也不可信：不能出现"显著但判定为红"这种自相矛盾的展示
        sig = np.zeros(sig.size, bool)
        q = np.ones(q.size, float)
        lvl_fdr = np.where(lvl_fdr >= 0, 0, lvl_fdr)
        n_sig = 0

    se_boot = res["se_w_boot"]
    rows = [{
        "feature": feature_names[i],
        "weight": float(res["w"][i]),
        "se": float(res["se_w"][i]),
        "se_boot": (None if se_boot is None else float(se_boot[i])),
        "z": float(z[i]),
        "p": float(p[i]),
        "q": float(q[i]),
        "sig": bool(sig[i]),
        "level": LEVEL_NAME[int(level[i])],
        "level_fdr": LEVEL_NAME[int(lvl_fdr[i])],
        "level_zh": LEVEL_ZH[int(lvl_fdr[i])],
        "n_needed": int(need[i]),
        "reachable": bool(need[i] < N_CAP) and int(lvl_fdr[i]) >= 0,
    } for i in range(w.size)]

    unstable_reason = None
    if res["n_neg"] > 0:
        unstable_reason = ("Hessian 有 %d 个负特征值：θ* 不在极大值点上，"
                           "sandwich 方差无意义" % res["n_neg"])
    elif res["cond_H"] > COND_LIMIT:
        unstable_reason = ("Hessian 条件数 %.2e：方向曲率接近 0，"
                           "该方向上的权重不可识别" % res["cond_H"])

    return {
        "rows": rows,
        "w": res["w"],
        "se_w": res["se_w"],
        "se_w_boot": se_boot,
        "z": z,
        "level": level,
        "level_fdr": lvl_fdr,
        "p": p,
        "q": q,
        "sig": sig,
        "fdr_alpha": float(fdr_alpha),
        "n_tests": int(w.size),
        "n_sig": int(n_sig),
        "rho_trace": res["rho_trace"],
        "cond_H": res["cond_H"],
        "n_neg": res["n_neg"],
        "unstable": res["unstable"],
        "identifiable": identifiable,
        "unstable_reason": unstable_reason,
        "separable": res["separable"],
        "nll_per_sample": res["nll_per_sample"],
        "n": n,
        "k": int(k),
        "target_se": float(target_se),
        "n_boot": int(n_boot),
    }
