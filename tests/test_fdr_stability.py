# -*- coding: utf-8 -*-
"""P1 自检：BH-FDR 多重校正 + 无标签可信度（分半迁移稳定性 + 置换零分布）。

跑法：
    <venv python> tests/test_fdr_stability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybridscope.diagnostics import bh_fdr, diagnose, two_sided_p      # noqa: E402
from hybridscope.stability import (                                     # noqa: E402
    fit_weights_unsupervised, label_free_credibility, stability_curve, null_curve,
    to_text)

OK = FAIL = 0


def check(name: str, cond: bool, info: str = ""):
    global OK, FAIL
    if cond:
        OK += 1
        print("  PASS  %s" % name)
    else:
        FAIL += 1
        print("  FAIL  %s   %s" % (name, info))


def synth(n: int = 120, m: int = 6, n_clust: int = 3, sep: float = 3.0,
          seed: int = 0) -> np.ndarray:
    """n_clust 个分离良好的高斯簇；sep 越大结构越强。"""
    rng = np.random.default_rng(seed)
    c = rng.normal(0, sep, size=(n_clust, m))
    lab = rng.integers(0, n_clust, n)
    return c[lab] + rng.normal(0, 1.0, size=(n, m))


print("== 1. BH-FDR 数值 ==")
p = np.array([0.01, 0.02, 0.03, 0.50, 0.60])
q, sig, n_sig = bh_fdr(p, 0.05)
# 手算：q_i = min_{j>=i} p_j·m/j → [0.05, 0.05, 0.05, 0.60, 0.60]
check("q 值单调化正确", np.allclose(q, [0.05, 0.05, 0.05, 0.60, 0.60]),
      str(np.round(q, 4)))
check("显著个数 = 3", n_sig == 3, str(n_sig))
check("q ≤ 1", bool(np.all(q <= 1.0)))
check("全大 p 时无显著", bh_fdr(np.array([0.4, 0.5, 0.6]), 0.05)[2] == 0)
check("全小 p 时全显著", bh_fdr(np.array([1e-6, 1e-5]), 0.05)[2] == 2)
check("双尾 p：z=1.96 → 0.05", abs(two_sided_p(np.array([1.96]))[0] - 0.05) < 1e-3,
      "%.4f" % two_sided_p(np.array([1.96]))[0])
check("BH 比 Bonferroni 宽松",
      bh_fdr(np.array([0.02, 0.03]), 0.05)[2] >= int(np.sum(np.array([0.02, 0.03]) < 0.025)))

print("== 2. 结构 vs 无结构：稳定性判别 ==")
# 判别力靠"分半迁移"：两个划分落在互不相交的样本上，才逼得出真实结构。
X_struct = synth(n=220, m=6, n_clust=3, sep=5.0, seed=1)
X_noise = np.random.default_rng(2).normal(0, 1, size=(220, 6))
w = np.full(6, 1.0 / 6)

obs_s = stability_curve(X_struct, None, w, k=5, n_boot=30, seed=3)
nul_s = null_curve(X_struct, None, w, k=5, n_perm=30, seed=3)
obs_n = stability_curve(X_noise, None, w, k=5, n_boot=30, seed=4)
nul_n = null_curve(X_noise, None, w, k=5, n_perm=30, seed=4)
print("     有结构 obs=%.3f null=%.3f | 无结构 obs=%.3f null=%.3f"
      % (obs_s.mean(), nul_s.mean(), obs_n.mean(), nul_n.mean()))
check("有结构时迁移 ARI 高于零分布", obs_s.mean() > nul_s.mean() + 0.10,
      "%.3f vs %.3f" % (obs_s.mean(), nul_s.mean()))
check("无结构时迁移 ARI 贴近零分布",
      abs(obs_n.mean() - nul_n.mean()) < 0.15,
      "%.3f vs %.3f" % (obs_n.mean(), nul_n.mean()))

rep_s = label_free_credibility(X_struct, None, w, k=5, n_boot=40, n_perm=60,
                               seed=5, n_boot_lofo=8)
rep_n = label_free_credibility(X_noise, None, w, k=5, n_boot=40, n_perm=60,
                               seed=6, n_boot_lofo=8)
print("     有结构 %s(严格 p=%.3f / 宽松 p=%.3f) | 无结构 %s(严格 p=%.3f / 宽松 p=%.3f)"
      % (rep_s["level"], rep_s["p_null"], rep_s["p_gauss"],
         rep_n["level"], rep_n["p_null"], rep_n["p_gauss"]))
# green 要求通过"严格对照"（各列独立、保留边际），是刻意设得很硬的门槛：
# 边际本身多峰时严格基线会偏高，此时只给 amber（判据来自宽松对照）。
check("有结构时至少判为勉强(amber/green)", rep_s["level"] in ("green", "amber"),
      "%s 严格p=%.3f 宽松p=%.3f" % (rep_s["level"], rep_s["p_null"], rep_s["p_gauss"]))
check("有结构时通过宽松对照（比无结构噪声强）", rep_s["p_gauss"] < 0.05,
      "%.3f" % rep_s["p_gauss"])
check("有结构时稳定性明显高于无结构", rep_s["stability"] > rep_n["stability"] + 0.3,
      "%.3f vs %.3f" % (rep_s["stability"], rep_n["stability"]))
check("无结构判为不可辨识(red)", rep_n["level"] == "red",
      "%s 严格p=%.3f 宽松p=%.3f" % (rep_n["level"], rep_n["p_null"], rep_n["p_gauss"]))
check("无结构不通过任一对照",
      rep_n["p_null"] >= 0.05 and rep_n["p_gauss"] >= 0.05,
      "%.3f / %.3f" % (rep_n["p_null"], rep_n["p_gauss"]))
check("严格对照基线不低于宽松对照基线（更保守）",
      rep_s["null_mean"] >= rep_s["gauss_mean"] - 0.05,
      "%.3f vs %.3f" % (rep_s["null_mean"], rep_s["gauss_mean"]))
check("留一表里每个特征都有 Δ", len(rep_s.get("lofo", [])) == 6,
      str(len(rep_s.get("lofo", []))))
check("留一按 Δ 降序排列",
      all((rep_s["lofo"][i]["drop"] or -9) >= (rep_s["lofo"][i + 1]["drop"] or -9)
          for i in range(len(rep_s["lofo"]) - 1)))
check("报告文本可生成", "无标签可信度" in to_text(rep_s))

print("== 3. 无标签拟合权重 ==")
res = fit_weights_unsupervised(X_struct, None, k=5, particles=10, iters=10, seed=0)
check("权重落在单纯形上", abs(res["w"].sum() - 1.0) < 1e-9 and np.all(res["w"] >= 0),
      str(np.round(res["w"], 3)))
check("轮廓系数有限", np.isfinite(res["silhouette"]), str(res["silhouette"]))
check("给出簇标签", len(res["labels"]) == len(X_struct))

print("== 4. diagnose 的 FDR 一致性 ==")
# 数值 Hessian 是有限差分算的，kNN 概率对 w 分段常数；
# 用带信息列与噪声列的合成数据（3 类）才不会得到全零的 H。
from knn_hybrid.estimability import make_d3_dataset                     # noqa: E402

d3 = make_d3_dataset(n=200, d_inf=3, d_noise=2, d_cat=2, n_clusters=3,
                     sep=1.5, seed=3)
Xn, Xc, yl = d3.X_num, d3.X_cat, np.asarray(d3.y, int)
names = list(d3.num_names) + list(d3.cat_names)
wl = np.full(Xn.shape[1] + Xc.shape[1], 1.0 / (Xn.shape[1] + Xc.shape[1]))
d = diagnose(Xn, Xc, yl, wl, k=5, fdr_alpha=0.05, n_boot=0, seed=0,
             feature_names=names)
rows = d["rows"]
check("每行都有 q / sig / level_fdr",
      all(("q" in r and "sig" in r and "level_fdr" in r) for r in rows))
check("显著 ⇒ 档位不是红/边界",
      all(r["level_fdr"] in ("green", "amber") for r in rows if r["sig"]),
      str([(r["feature"], r["sig"], r["level_fdr"]) for r in rows]))
check("不显著 ⇒ 档位为红或边界",
      all(r["level_fdr"] in ("red", "boundary") for r in rows if not r["sig"]),
      str([(r["feature"], r["sig"], r["level_fdr"]) for r in rows]))
check("n_sig 与 sig 计数一致",
      d["n_sig"] == int(sum(bool(r["sig"]) for r in rows)),
      "%d vs %d" % (d["n_sig"], sum(bool(r["sig"]) for r in rows)))
check("p 值单调对应 q 值排序",
      bool(np.all(np.diff(np.array([r["p"] for r in rows])) *
                  np.diff(np.array([r["q"] for r in rows])) >= -1e-12)))
check("不可识别时不会有显著项",
      (d["identifiable"] and d["n_sig"] >= 0) or d["n_sig"] == 0)
check("梯度 bootstrap 关闭时 se_boot 为空",
      all(r["se_boot"] is None for r in rows))

print("== 5. 梯度 bootstrap（J 重抽） ==")
d2 = diagnose(Xn, Xc, yl, wl, k=5, n_boot=30, seed=0, feature_names=names)
sb = np.array([r["se_boot"] for r in d2["rows"]], float)
se = np.array([r["se"] for r in d2["rows"]], float)
act = se > 0          # 边界权重（w=0）的 SE 为 0，不参与比较
ratio = sb[act] / se[act]
check("bootstrap SE 与 sandwich SE 同量级（非边界权重）",
      bool(np.all(np.isfinite(sb)) and np.all(np.abs(ratio - 1.0) < 0.5)),
      "比值 %s" % np.round(ratio, 3))

print("\n通过 %d 项，失败 %d 项" % (OK, FAIL))
sys.exit(1 if FAIL else 0)
