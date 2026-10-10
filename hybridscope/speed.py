# -*- coding: utf-8 -*-
"""算力预算：在「跑得完」和「结论站得住」之间做显式取舍。

大表跑不动是这套方法落地时最先撞上的墙：PSO 每一次评估都要重算一遍 n×n
的加权距离矩阵，代价随 n² 增长；稳定性与置换零分布又要各跑几十次聚类。
n 从 200 涨到 400，实测 PSO 耗时从 1.2 s 涨到 7.8 s——不加约束地往上堆样本，
一次分析就要跑到十几分钟，实际没人会等。

本模块不发明近似算法（那会改变结论，得不偿失），只做三件可审计的事：
  1) 开工前按实测系数预估耗时与内存，给出「要不要降预算」的建议；
  2) 按 mode 给一套明确的预算：auto 随 n 缩放、fast 先求跑完、
     thorough 用于最终定稿；
  3) 把实际用了多少预算写进结果（快照 config），让读报告的人知道
     这个结论是花了多少算力换来的。

降预算的代价必须说清楚：
  · 减少 PSO 粒子/迭代 → 权重是随机搜索的解，预算小则解更粗，
    表现为同一批数据换种子重跑的离散度变大；
  · 稳定性检验改用子样本 → 观测值与零分布必须用同一个子样本规模，
    否则两边的方差不可比，z 值会失真。子采样只影响精度，不影响无偏性。
"""

from __future__ import annotations

import math

# 实测标定：n=200 → 单次 PSO 评估约 4 ms，n=400 → 约 26 ms。
# 拟合指数 2.5（介于 n² 的距离矩阵与 n²log n 的近邻排序之间）。
_REF_N = 200
_REF_EVAL = 0.004          # 秒 / 次评估（n=200, m=6, d_cat=0, rounds=2）
_REF_M = 6
_EXP = 2.5


def eval_seconds(n: int, m: int, d_cat: int = 0, rounds: int = 2,
                 ref_eval: float = _REF_EVAL) -> float:
    """单次 PSO 评估的耗时估计（秒）。"""
    n = max(int(n), 2)
    load = max(1, (int(m) + int(d_cat))) / float(_REF_M)
    return ref_eval * load * (rounds / 2.0) * (n / float(_REF_N)) ** _EXP


def dist_matrix_bytes(n: int, d_cat: int = 0, low_memory: bool = True) -> float:
    """距离矩阵（含类别指示张量）的内存占用估计（字节）。

    E 张量是 n×n×d_cat，是大表最先爆掉的地方；low_memory 改成逐列累加后
    峰值从 O(n²·d_cat) 降到 O(n²)。
    """
    n = max(int(n), 1)
    base = n * n * 8.0                       # 数值距离矩阵
    cat = 0.0 if low_memory else n * n * max(int(d_cat), 0) * 8.0
    return base + cat


def cost_estimate(n: int, m: int, d_cat: int = 0, particles: int = 15,
                  iters: int = 20, rounds: int = 2, stab_boot: int = 60,
                  stab_perm: int = 60, n_boot: int = 200,
                  run_stability: bool = True, lofo_features: int = 0) -> dict:
    """一次完整分析的耗时/内存预估。量级准确即可，用来决定预算。"""
    t_eval = eval_seconds(n, m, d_cat, rounds)
    t_pso = t_eval * max(int(particles), 1) * max(int(iters), 1)
    t_stab = 0.0
    if run_stability:
        # 稳定性：每次分半/置换都要重跑一次聚类（≈ 一次评估 + 轮廓系数）
        per = t_eval * 1.5
        t_stab = per * (max(int(stab_boot), 0) + max(int(stab_perm), 0))
        t_stab += per * max(int(lofo_features), 0) * max(1, min(n_boot, 15))
    t_boot = 0.0
    if n_boot:
        # 梯度 bootstrap 只重抽得分矩阵，比一次评估便宜得多
        t_boot = max(int(n_boot), 0) * t_eval * 0.05
    total = t_pso + t_stab + t_boot
    mem = dist_matrix_bytes(n, d_cat, low_memory=True)
    return {
        "n": int(n), "m": int(m), "d_cat": int(d_cat),
        "seconds_pso": t_pso, "seconds_stability": t_stab,
        "seconds_bootstrap": t_boot, "seconds_total": total,
        "minutes_total": total / 60.0,
        "peak_bytes": mem,
        "peak_mb": mem / 1024.0 / 1024.0,
    }


def budget(n: int, m: int = 6, d_cat: int = 0, mode: str = "auto",
           run_stability: bool = True) -> dict:
    """按规模给一套预算。

    auto     随 n 缩放：小表不动，大表逐档压缩，尽量保住结论精度
    fast     先求跑完：粒子/迭代、稳定性次数都压到最小可用
    thorough 最终定稿用：不压缩，但会先告诉你要多久
    """
    n = max(int(n), 1)
    m = max(int(m), 1)
    particles, iters = 15, 20
    stab_boot, stab_perm, n_boot = 60, 60, 200
    subsample = None
    rounds = 2

    if mode == "thorough":
        return {"mode": mode, "pso_particles": particles, "pso_iters": iters,
                "contribution_rounds": rounds, "stab_boot": stab_boot,
                "stab_perm": stab_perm, "n_boot": n_boot, "subsample_n": None,
                "run_stability": run_stability, "low_memory": False}
    if mode == "fast":
        particles, iters = 8, 10
        stab_boot, stab_perm, n_boot = 20, 20, 60
        rounds = 1
        subsample = 600 if n > 600 else None
        return {"mode": mode, "pso_particles": particles, "pso_iters": iters,
                "contribution_rounds": rounds, "stab_boot": stab_boot,
                "stab_perm": stab_perm, "n_boot": n_boot, "subsample_n": subsample,
                "run_stability": run_stability, "low_memory": True}

    # auto：按预估耗时逐档降，而不是按 n 硬切——同样的 n，特征多就更慢
    est = cost_estimate(n, m, d_cat, particles, iters, 2,
                        stab_boot, stab_perm, n_boot, run_stability, m)
    if est["minutes_total"] <= 1.0:
        pass                                             # 一分钟以内，不动
    elif est["minutes_total"] <= 4.0:
        stab_boot, stab_perm, n_boot = 40, 40, 120
        subsample = 1200 if n > 1200 else None
    elif est["minutes_total"] <= 12.0:
        particles, iters = 12, 15
        stab_boot, stab_perm, n_boot = 30, 30, 80
        subsample = 1000 if n > 1000 else None
        low = True
    else:
        particles, iters = 10, 12
        rounds = 1
        stab_boot, stab_perm, n_boot = 20, 20, 50
        subsample = 800 if n > 800 else None
    return {"mode": "auto", "pso_particles": particles, "pso_iters": iters,
            "contribution_rounds": rounds, "stab_boot": stab_boot,
            "stab_perm": stab_perm, "n_boot": n_boot,
            "subsample_n": subsample, "run_stability": run_stability,
            "low_memory": True,
            "note": "按预估 %.1f 分钟降档" % est["minutes_total"]}


def advice(n: int, m: int = 6, d_cat: int = 0, particles: int = 15,
           iters: int = 20, stab_boot: int = 60, stab_perm: int = 60,
           n_boot: int = 200, run_stability: bool = True) -> str:
    """给用户的一段话：现在这套设置要跑多久、卡在哪、建议怎么调。"""
    est = cost_estimate(n, m, d_cat, particles, iters, 2, stab_boot, stab_perm,
                        n_boot, run_stability, m)
    L = ["算力预估：n=%d，特征 %d（类别 %d）" % (n, m, d_cat)]
    L.append("  PSO %s ｜ 稳定性 %s ｜ bootstrap %s ｜ 合计 %s"
             % (fmt_seconds(est["seconds_pso"]),
                fmt_seconds(est["seconds_stability"]),
                fmt_seconds(est["seconds_bootstrap"]),
                fmt_seconds(est["seconds_total"])))
    L.append("  距离矩阵峰值内存 %.1f MB" % est["peak_mb"])
    worst = max(("PSO", est["seconds_pso"]),
                ("稳定性与零分布", est["seconds_stability"]),
                ("bootstrap", est["seconds_bootstrap"]), key=lambda x: x[1])
    L.append("  主要开销：%s" % worst[0])
    if est["seconds_total"] > 300:
        b = budget(n, m, d_cat, "fast", run_stability)
        fast = cost_estimate(n, m, d_cat, b["pso_particles"], b["pso_iters"],
                             b["contribution_rounds"], b["stab_boot"],
                             b["stab_perm"], b["n_boot"], run_stability, m)
        L.append("  超过 5 分钟，建议 --speed fast（约 %s）或 --max-n 800 子采样；"
                 "代价是权重的随机搜索更粗、可信度检验精度下降。"
                 % fmt_seconds(fast["seconds_total"]))
    if est["peak_mb"] > 800:
        L.append("  内存吃紧（%.0f MB）：已按逐列累加算类别距离，若仍不够就先降 --max-n。"
                 % est["peak_mb"])
    return "\n".join(L)


def subsample_indices(n: int, cap: int | None, seed: int = 0) -> np.ndarray | None:
    """稳定性检验用的子样本下标；cap 为空或 n<=cap 时返回 None（用全样本）。"""
    import numpy as np
    if not cap or n <= int(cap):
        return None
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=int(cap), replace=False))


def fmt_seconds(s: float) -> str:
    if not math.isfinite(s):
        return "n/a"
    if s < 60:
        return "%.1f s" % s
    if s < 3600:
        return "%.1f min" % (s / 60.0)
    return "%.1f h" % (s / 3600.0)
