# -*- coding: utf-8 -*-
"""无标签可信度：分半迁移稳定性 + 置换零分布 + 逐特征留一检验。

没有标签时，CompositeLikelihood 那条 sandwich 路走不通（它需要 y）。
这里换一条不依赖标签的证据链：

    1. 稳定性（transfer ARI）：把样本随机分成两半 A/B，各自用权重 w 聚类；
       再用加权距离把 B 中每个点"迁移"到最近的 A 点的簇标签上，
       算迁移标签与 B 自己聚类结果之间的 ARI。ARI 是机会校正的，
       所以"没有结构时两套划分互不相关"会自动落到 0 附近。
    2. 零分布：把每一列独立打乱 —— 保留各列边际分布，只抹掉列间依赖
       （即"各考核项相互独立"这一零假设）。同一套流程再跑一遍，
       得到"没有列间结构时这个流程也能迁移出多少"的基线。
    3. z 与 p：稳定性在零分布里的位置。z 高 = 结构不是流程假象。
    4. 逐特征留一：把第 j 个特征权重置 0 后稳定性掉多少。
       掉得多 = 该特征在撑着这套分型。

为什么不用"子抽样 vs 全样本"这个更常见的稳定性定义：实测下来它对
"有没有结构"几乎不敏感（真实成绩 0.83，列打乱后仍有 0.86）——
它衡量的是流程可复现性，不是结构显著性。分半迁移把两个划分放在
互不相交的样本上，才逼出真实的判别力（合成 3 簇 0.78 vs 打乱 −0.01）。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

from knn_hybrid.model import WeightedDistance, mutual_knn_clustering
from knn_hybrid.pso import PSO, _softmax

MIN_N = 20   # 分半迁移需要的最小样本量


# ---------- 基础算子 ---------- #

def _as(X_num, X_cat):
    X_num = np.asarray(X_num, float)
    X_cat = (np.asarray(X_cat, int) if X_cat is not None
             else np.zeros((X_num.shape[0], 0), int))
    return X_num, X_cat


def cluster_with_w(X_num, X_cat, w, k: int = 5) -> np.ndarray:
    """给定权重做互 kNN 聚类，返回标签。"""
    X_num, X_cat = _as(X_num, X_cat)
    D = WeightedDistance(X_num, X_cat)(np.asarray(w, float), np.ones(X_num.shape[0]))
    return mutual_knn_clustering(D, k)


def _dist_matrix(X_num, X_cat, w):
    X_num, X_cat = _as(X_num, X_cat)
    return WeightedDistance(X_num, X_cat)(np.asarray(w, float),
                                          np.ones(X_num.shape[0]))


def _silhouette(X_num, X_cat, w, k: int, labels=None):
    """加权距离下的轮廓系数；簇数不合法时返回 nan。"""
    X_num, X_cat = _as(X_num, X_cat)
    n = X_num.shape[0]
    D = _dist_matrix(X_num, X_cat, w)
    if labels is None:
        labels = mutual_knn_clustering(D, k)
    n_cl = len(np.unique(labels))
    if n_cl < 2 or n_cl > n - 1:
        return np.nan, labels, D
    return float(silhouette_score(D, labels, metric="precomputed")), labels, D


def fit_weights_unsupervised(X_num, X_cat, k: int = 5, particles: int = 12,
                             iters: int = 12, seed: int = 0) -> dict:
    """无标签时拟合权重：以轮廓系数为目标做一次 PSO。

    有标签时目标是分类误差，无标签时只能退到"簇内紧、簇间开"。
    同一单纯形上的搜索，只是目标函数不同 —— w 的含义不变，
    后面的稳定性检验对同一个 w 做。
    """
    X_num, X_cat = _as(X_num, X_cat)
    m = X_num.shape[1] + X_cat.shape[1]

    def fitness(w):
        s, _, _ = _silhouette(X_num, X_cat, w, k)
        return -s if np.isfinite(s) else 1.0  # nan → 最差

    pso = PSO(dim=m, n_particles=particles, iters=iters, seed=seed)
    x, best_f, hist = pso.optimize(fitness, encoding="weight")
    w = _softmax(x)
    sil, labels, _ = _silhouette(X_num, X_cat, w, k)
    return {"w": w, "silhouette": sil, "objective": -float(best_f),
            "labels": labels, "history": [float(h) for h in hist],
            "n_clusters": int(len(np.unique(labels)))}


# ---------- 稳定性和零分布 ---------- #

def _permute_columns(X, rng):
    """逐列独立打乱：保留每列边际分布，抹掉列间依赖。"""
    X = np.asarray(X).copy()
    for j in range(X.shape[1]):
        rng.shuffle(X[:, j])
    return X


def transfer_ari(X_num, X_cat, w, k: int, rng, frac: float = 0.5,
                 D_all=None) -> float:
    """一次分半迁移：返回 ARI(B 自己的聚类, 由 A 迁移过来的标签)。"""
    X_num, X_cat = _as(X_num, X_cat)
    n = X_num.shape[0]
    if D_all is None:
        D_all = _dist_matrix(X_num, X_cat, w)
    idx = rng.permutation(n)
    h = int(round(frac * n))
    h = int(np.clip(h, MIN_N // 2, n - MIN_N // 2))
    A, B = idx[:h], idx[h:]
    labA = cluster_with_w(X_num[A], X_cat[A] if X_cat.shape[1] else None, w, k)
    labB = cluster_with_w(X_num[B], X_cat[B] if X_cat.shape[1] else None, w, k)
    pred = labA[np.argmin(D_all[np.ix_(B, A)], axis=1)]
    return float(adjusted_rand_score(labB, pred))


def stability_curve(X_num, X_cat, w, k: int = 5, n_boot: int = 60,
                    frac: float = 0.5, seed: int = 0) -> np.ndarray:
    """n_boot 次分半迁移 ARI。"""
    X_num, X_cat = _as(X_num, X_cat)
    if X_num.shape[0] < MIN_N:
        return np.array([], float)
    D_all = _dist_matrix(X_num, X_cat, w)
    rng = np.random.default_rng(seed)
    return np.array([transfer_ari(X_num, X_cat, w, k, rng, frac, D_all)
                     for _ in range(int(n_boot))], float)


def null_curve(X_num, X_cat, w, k: int = 5, n_perm: int = 60, frac: float = 0.5,
               seed: int = 0, kind: str = "perm") -> np.ndarray:
    """零分布下的迁移 ARI。

    kind="perm"：逐列独立打乱 —— 保留各列边际分布，只抹掉列间依赖。
        这是**严格**零假设：边际本身多峰时（比如成绩天然分档、设备天然分工况），
        打乱后的点云仍有团块，基线会偏高，所以这个零分布很保守、功效偏低。
    kind="gauss"：每列用同均值同方差的正态替代（类别列按经验频率重抽）。
        这是**宽松**零假设：只保留一二阶矩，连边际形状都不要了。
    两个都给：p_perm 显著是最硬的证据；只有 p_gauss 显著说明"比无结构噪声强，
    但还没强过各列边际自身带来的团块"。
    """
    X_num, X_cat = _as(X_num, X_cat)
    if X_num.shape[0] < MIN_N:
        return np.array([], float)
    rng = np.random.default_rng(seed + 7717)
    out = np.zeros(int(n_perm), float)
    mu = X_num.mean(axis=0)
    sd = np.where(X_num.std(axis=0) > 1e-12, X_num.std(axis=0), 1.0)
    for b in range(int(n_perm)):
        if kind == "gauss":
            Xn_p = rng.normal(mu, sd, size=X_num.shape)
        else:
            Xn_p = _permute_columns(X_num, rng)
        Xc_p = _permute_columns(X_cat, rng) if X_cat.shape[1] else X_cat
        out[b] = transfer_ari(Xn_p, Xc_p, w, k, rng, frac)
    return out


def _null_stats(obs_mean: float, null: np.ndarray) -> tuple[float, float, float]:
    """返回 (零分布均值, z, 经验 p)。"""
    if null.size == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(null))
    s = float(np.std(null, ddof=1)) if null.size > 1 else np.nan
    if not (np.isfinite(s) and s > 1e-12):
        return m, np.nan, np.nan
    z = (obs_mean - m) / s
    p = float((np.sum(null >= obs_mean) + 1.0) / (null.size + 1.0))
    return m, float(z), p


def leave_one_feature_out(X_num, X_cat, w, k: int = 5, n_boot: int = 15,
                          frac: float = 0.5, seed: int = 0,
                          feature_names=None, base_stability=None) -> list[dict]:
    """逐特征置零后重算稳定性：掉得越多，该特征越在撑着这套分型。"""
    X_num, X_cat = _as(X_num, X_cat)
    m = w.size
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(m)]
    rows = []
    for j in range(m):
        w2 = w.copy()
        w2[j] = 0.0
        s = w2.sum()
        w2 = np.full(m, 1.0 / m) if s <= 1e-12 else w2 / s
        aris = stability_curve(X_num, X_cat, w2, k, n_boot=n_boot, frac=frac,
                               seed=seed + 101 * (j + 1))
        st = float(np.mean(aris)) if aris.size else np.nan
        drop = (base_stability - st) if (base_stability is not None
                                         and np.isfinite(st)) else np.nan
        rows.append({
            "feature": feature_names[j],
            "weight": float(w[j]),
            "stability_without": st,
            "drop": drop,
            "sd": float(np.std(aris)) if aris.size else np.nan,
        })
    return rows


LEVEL_NAME = {2: "green", 1: "amber", 0: "red"}
LEVEL_ZH = {2: "可信", 1: "勉强", 0: "不可辨识"}


def label_free_credibility(X_num, X_cat, w, k: int = 5, n_boot: int = 60,
                           n_perm: int = 50, frac: float = 0.5, seed: int = 0,
                           feature_names=None, do_lofo: bool = True,
                           n_boot_lofo: int = 15) -> dict:
    """一次调用给出无标签情形下的完整可信度结论（双零分布）。"""
    X_num, X_cat = _as(X_num, X_cat)
    w = np.asarray(w, float)
    n = X_num.shape[0]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(w.size)]

    sil, ref_labels, _ = _silhouette(X_num, X_cat, w, k)
    obs = stability_curve(X_num, X_cat, w, k, n_boot=n_boot, frac=frac, seed=seed)
    nlp = null_curve(X_num, X_cat, w, k, n_perm=n_perm, frac=frac, seed=seed,
                     kind="perm")
    nlg = null_curve(X_num, X_cat, w, k, n_perm=n_perm, frac=frac,
                     seed=seed + 313, kind="gauss")

    stab = float(np.mean(obs)) if obs.size else np.nan
    nl_mean, z, p_emp = _null_stats(stab, nlp)          # 严格：各列独立
    ng_mean, z_g, p_g = _null_stats(stab, nlg)          # 宽松：正态替代

    if np.isfinite(z) and np.isfinite(p_emp):
        if p_emp < 0.05 and stab >= 0.50:
            lvl, basis = 2, "perm"
        elif (np.isfinite(p_g) and p_g < 0.05 and stab >= 0.50) or p_emp < 0.05:
            lvl, basis = 1, ("gauss" if not (p_emp < 0.05) else "perm")
        else:
            lvl, basis = 0, "none"
    else:
        lvl, basis = 0, "none"

    out = {
        "mode": "label-free",
        "metric": "transfer ARI (split-half, weighted 1-NN transfer)",
        "null_model": "perm = 逐列打乱（保留边际，严格）; gauss = 同均值方差正态替代（宽松）",
        "n": int(n),
        "k": int(k),
        "n_clusters": int(len(np.unique(ref_labels))),
        "silhouette": sil,
        "stability": stab,
        "stability_sd": float(np.std(obs)) if obs.size else np.nan,
        "stability_curve": obs.tolist(),
        "null_mean": nl_mean,          # 主对照：严格零分布
        "null_sd": float(np.std(nlp, ddof=1)) if nlp.size > 1 else np.nan,
        "null": nlp.tolist(),
        "gauss": nlg.tolist(),         # 宽松零分布的原始样本，供报告画图
        "z": float(z) if np.isfinite(z) else None,
        "p_null": p_emp,
        "gauss_mean": ng_mean,
        "gauss_sd": float(np.std(nlg, ddof=1)) if nlg.size > 1 else np.nan,
        "z_gauss": float(z_g) if np.isfinite(z_g) else None,
        "p_gauss": p_g,
        "basis": basis,
        "n_boot": int(n_boot),
        "n_perm": int(n_perm),
        "level": LEVEL_NAME[int(lvl)],
        "level_zh": LEVEL_ZH[int(lvl)],
        "verdict": _verdict(lvl, stab, p_emp, z, nl_mean, p_g, z_g, ng_mean, basis),
    }
    if do_lofo and w.size > 1 and n >= MIN_N:
        rows = leave_one_feature_out(X_num, X_cat, w, k, n_boot=n_boot_lofo,
                                     frac=frac, seed=seed,
                                     feature_names=feature_names,
                                     base_stability=stab)
        rows.sort(key=lambda r: -(r["drop"] if np.isfinite(r["drop"]) else -9.0))
        out["lofo"] = rows
        out["top_features"] = [r["feature"] for r in rows[:3]
                               if np.isfinite(r["drop"]) and r["drop"] > 0.01]
    return out


def _verdict(lvl, stab, p, z, nl_mean, p_g=None, z_g=None, ng_mean=np.nan,
             basis="none") -> str:
    base = ("分半迁移一致性 %.2f" % stab if np.isfinite(stab) else "分半迁移一致性 n/a")
    tail_perm = "严格对照基线 %.2f，z=%s，p=%s" % (
        nl_mean, "%.1f" % z if np.isfinite(z) else "n/a",
        "%.3f" % p if np.isfinite(p) else "n/a")
    tail_g = "宽松对照基线 %.2f，z=%s，p=%s" % (
        ng_mean, "%.1f" % z_g if np.isfinite(z_g) else "n/a",
        "%.3f" % p_g if np.isfinite(p_g) else "n/a")
    if lvl == 2:
        return ("%s：显著高于%s —— 这套分型反映的是真实结构，"
                "不只是聚类的顺势而为，也不只是各列边际自身造成的团块。可以基于此下结论。"
                % (base, tail_perm))
    if lvl == 1:
        if basis == "gauss":
            return ("%s：严格对照下不显著（%s），但显著高于%s —— "
                    "比无结构噪声强，还不排除「各列边际自身就带团块」这一解释。"
                    "可以做探索性分组，不要据此给个体下结论。"
                    % (base, tail_perm, tail_g))
        return ("%s：通过了严格对照（%s），但一致性本身偏低（<0.50）—— "
                "结构方向可信、粒度不够稳。建议调大 k 近邻或补样本再判。"
                % (base, tail_perm))
    return ("%s：与%s 区分不开；宽松对照下也%s —— "
            "这张表在该权重下没有可辨识的簇结构，分型是硬切出来的。"
            "不要基于它给个体贴标签。"
            % (base, tail_perm,
               "不显著（p=%s）" % ("%.3f" % p_g if np.isfinite(p_g) else "n/a")
               if not (np.isfinite(p_g) and p_g < 0.05) else "显著，但粒度不够"))


def to_text(rep: dict) -> str:
    """把无标签可信度报告压成一段可读文本（供 CLI / 报告使用）。"""
    L = ["无标签可信度（分半迁移稳定性 + 双零分布）"]
    L.append("  样本 %d，簇数 %d，轮廓系数 %s"
             % (rep["n"], rep["n_clusters"],
                "%.3f" % rep["silhouette"] if np.isfinite(rep["silhouette"]) else "n/a"))
    L.append("  迁移一致性 ARI = %.3f ± %.3f（%d 次分半）"
             % (rep["stability"], rep["stability_sd"], rep["n_boot"]))
    L.append("  严格零分布（列打乱，保留边际）= %.3f ± %.3f → z=%s，p=%s"
             % (rep["null_mean"], rep["null_sd"],
                "%.2f" % rep["z"] if rep["z"] is not None else "n/a",
                "%.3f" % rep["p_null"] if np.isfinite(rep["p_null"]) else "n/a"))
    L.append("  宽松零分布（同均值方差正态）= %.3f ± %.3f → z=%s，p=%s"
             % (rep.get("gauss_mean", np.nan), rep.get("gauss_sd", np.nan),
                "%.2f" % rep["z_gauss"] if rep.get("z_gauss") is not None else "n/a",
                "%.3f" % rep["p_gauss"] if np.isfinite(rep.get("p_gauss", np.nan)) else "n/a"))
    L.append("  → %s（%s），判据来自%s" % (rep["level"], rep["level_zh"],
                                          {"perm": "严格对照", "gauss": "宽松对照",
                                           "none": "未通过任一对照"}.get(rep.get("basis"), "-")))
    L.append("  " + rep["verdict"])
    if rep.get("lofo"):
        L.append("  逐特征留一（去掉后稳定性掉得越多＝越在撑着分型）：")
        for r in rep["lofo"]:
            L.append("    %-14s w=%.3f  去掉后 %.3f  Δ=%s"
                     % (r["feature"], r["weight"], r["stability_without"],
                        "%+.3f" % r["drop"] if np.isfinite(r["drop"]) else "n/a"))
    if rep.get("top_features"):
        L.append("  撑着分型的特征：%s" % "、".join(rep["top_features"]))
    return "\n".join(L)
