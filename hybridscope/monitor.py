# -*- coding: utf-8 -*-
"""跨批次对比：把「权重变了」翻译成「真漂移 / 只是噪声 / 数据先变了」。

一次分析本身没有价值，对比才有。但两次跑出来的权重几乎不可能完全一样，
所以真正要问的不是「变没变」，而是「变化量相对它的标准误有多大」。

两个独立批次的权重差 Δw_i = w_b,i − w_a,i，其标准误
    se(Δw_i) = sqrt(se_a,i² + se_b,i²)
（独立样本方差相加；若两批有重叠样本则该式偏保守，即更容易判为「只是噪声」）。
逐特征给 z 与 BH-FDR 校正后的 q，再给一个整体卡方检验
    T = Σ Δw_i² / se(Δw_i)²,  df = m − 1
（减 1 是因为权重落在单纯形上，ΣΔw = 0，只有 m−1 个自由度）。

三种结论必须先分清，否则一定误读：
  1) 数据指纹相同 → Δw 只可能来自优化器随机性，不是真实漂移；
  2) 数据变了、权重没显著变 → 结论稳，但要看数据侧漂移清单（换了好教的学生？换了多少分制？）；
  3) 数据变了、权重显著变 → 真漂移，逐特征列出谁变了、往哪变。
"""

from __future__ import annotations

import datetime as _dt
import json

import numpy as np

try:                                    # KS 检验只在 scipy 可用时启用
    from scipy import stats as _st
except Exception:                       # pragma: no cover
    _st = None


def _z_p(z: np.ndarray) -> np.ndarray:
    """双侧正态 p 值；scipy 缺失时用误差函数近似。"""
    z = np.asarray(z, float)
    if _st is not None:
        return 2.0 * _st.norm.sf(np.abs(z))
    from math import erfc, sqrt
    return np.array([erfc(abs(v) / sqrt(2.0)) for v in z], float)


def bh_fdr(p: np.ndarray, alpha: float = 0.05):
    """与 diagnostics.bh_fdr 相同口径的 BH 校正，这里独立实现避免循环依赖。"""
    p = np.asarray(p, float)
    m = p.size
    if m == 0:
        return np.array([], float), np.array([], bool), 0
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1.0)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m, float)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q, q <= float(alpha), int(np.sum(q <= float(alpha)))


# ---------------------------------------------------------------- 快照

def snapshot(est, df=None, name: str | None = None, source: str | None = None,
             notes: str | None = None, schema=None, profile: str | None = None,
             aliases: dict | None = None) -> dict:
    """把一次拟合结果封成可存档、可对比的快照。

    schema：这次分析用的字段映射（hybridscope.schema.Schema 或 dict）。带上它，
    跨批次对比时才知道哪些 canonical 字段是声明为可比的、哪些只是同名不同义。
    aliases：canonical 名 → 建模时实际用的列名（建模常用中文列名，映射用英文名）。
    """
    from hybridscope import __version__
    from hybridscope.io import data_fingerprint

    fp = None
    if df is not None:
        try:
            fp = data_fingerprint(df)
        except Exception as exc:      # 指纹失败不该让整个分析挂掉，但要留痕
            fp = {"error": str(exc)}
    if fp is None or "error" in fp:
        fp = getattr(est, "data_fingerprint_", None)

    summ = est.summary()
    dg = getattr(est, "diagnostics_", None) or {}
    se = None
    if "se_w" in dg:
        se = np.asarray(dg["se_w"], float)
        if se.size != np.asarray(est.w_, float).size:
            se = None
    cluster_sizes = []
    if getattr(est, "labels_", None) is not None:
        _, cnt = np.unique(np.asarray(est.labels_), return_counts=True)
        cluster_sizes = [int(c) for c in cnt]

    return {
        "schema": "hybridscope-snapshot/1",
        "name": name,
        "source": source,
        "notes": notes,
        "field_schema": _schema_meta(schema, profile, aliases),
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "version": __version__,
        "config": {
            "k": getattr(est, "k_", getattr(est, "k", None)),
            "seed": getattr(est, "seed", None),
            "denoise": summ.get("denoise"),
            "pso_particles": getattr(est, "pso_particles", None),
            "pso_iters": getattr(est, "pso_iters", None),
            "fdr_alpha": getattr(est, "fdr_alpha", None),
            "speed": (getattr(est, "speed_", None) or {}).get("mode",
                                                             getattr(est, "speed", None)),
            "speed_budget": getattr(est, "speed_", None),
        },
        "fingerprint": fp,
        "n": int(summ.get("n", 0)),
        "feature_names": [str(f) for f in getattr(est, "feature_names_", [])],
        "w": [float(x) for x in np.asarray(est.w_, float)] if est.w_ is not None else None,
        "se_w": [float(x) for x in se] if se is not None else None,
        "metrics": summ.get("metrics"),
        "identifiable": dg.get("identifiable"),
        "n_clusters": len(cluster_sizes),
        "cluster_sizes": cluster_sizes,
        "stability": summ.get("stability"),
        "diagnostics": summ.get("diagnostics"),
    }


def _schema_meta(schema, profile: str | None = None,
                 aliases: dict | None = None) -> dict | None:
    """把字段映射压成快照里的一小段元信息（不重复存整个映射）。"""
    if schema is None:
        return None
    d = schema.to_dict() if hasattr(schema, "to_dict") else dict(schema)
    comp, uncomp = [], []
    for f in d.get("fields", []):
        if f.get("role") != "feature":
            continue
        (comp if f.get("comparable", True) else uncomp).append(f.get("name"))
    return {"id": d.get("name"), "version": str(d.get("version", "1")),
            "profile": profile, "comparable": comp, "not_comparable": uncomp,
            "aliases": dict(aliases or {})}


def _blocked_from_schema(fs: dict) -> dict[str, str]:
    """把「映射声明不可比」的 canonical 名翻译成建模用的列名。"""
    al = fs.get("aliases") or {}
    out = {}
    for n in fs.get("not_comparable") or []:
        col = al.get(n, n)
        out[str(col)] = ("字段映射把 %s 声明为不可跨批次比较（同名不同义）" % n)
    return out


def save_snapshot(snap: dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2, default=_jsonable)
    return path


def load_snapshot(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(float(o)) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------- 数据侧漂移

def data_drift(a: dict, b: dict, smd_flag: float = 0.20,
               ks_flag: float = 0.20, tvd_flag: float = 0.15) -> dict:
    """比较两份数据指纹：列是否一致、边际分布漂了多少。"""
    fa, fb = a.get("fingerprint"), b.get("fingerprint")
    out = {
        "has_fingerprint": bool(fa and fb and fa.get("columns") and fb.get("columns")),
        "same_data_rows": False,
        "same_data_pool": False,
        "columns_only_in_a": [], "columns_only_in_b": [],
        "n_rows": {"a": a.get("n"), "b": b.get("n")},
        "numeric": [], "categorical": [],
        "n_flagged": 0,
        "flagged": [],
    }
    if not (fa and fb and fa.get("columns") and fb.get("columns")):
        why = []
        for tag, f in (("A", fa), ("B", fb)):
            if not f:
                why.append("%s 缺指纹" % tag)
            elif not f.get("columns"):
                why.append("%s 指纹生成失败（%s）" % (tag, f.get("error", "未知")))
        out["note"] = ("至少一份快照没有可用的数据指纹（%s），数据侧漂移无法核对；"
                       "权重侧比较仍然有效。" % "，".join(why))
        return out

    from hybridscope.io import fingerprint_equal
    out["same_data_rows"] = fingerprint_equal(fa, fb, pool=False)
    out["same_data_pool"] = fingerprint_equal(fa, fb, pool=True)

    ca, cb = set(fa.get("columns", [])), set(fb.get("columns", []))
    out["columns_only_in_a"] = sorted(ca - cb)
    out["columns_only_in_b"] = sorted(cb - ca)

    na, nb = fa.get("numeric", {}), fb.get("numeric", {})
    for c in sorted(ca & cb):
        if c not in na or c not in nb:
            continue
        sa, sb = na[c], nb[c]
        if sa.get("mean") is None or sb.get("mean") is None:
            continue
        sd_a = float(sa.get("std") or 0.0)
        sd_b = float(sb.get("std") or 0.0)
        pooled = np.sqrt((sd_a ** 2 + sd_b ** 2) / 2.0)
        d = float(sb["mean"] - sa["mean"])
        smd = d / pooled if pooled > 1e-12 else (0.0 if abs(d) < 1e-12 else np.inf)
        row = {"column": c, "mean_a": float(sa["mean"]), "mean_b": float(sb["mean"]),
               "delta": d, "std_a": sd_a, "std_b": sd_b, "smd": float(smd),
               "ks": None, "ks_p": None}
        if _st is not None and "q05" in sa and sa.get("q05") is not None:
            # 没有原始行，用五分位点构成的经验 CDF 近似 KS 距离（保守）
            ga = [sa[k] for k in ("q05", "q25", "q50", "q75", "q95")]
            gb = [sb[k] for k in ("q05", "q25", "q50", "q75", "q95")]
            row["ks"] = float(max(abs(x - y) for x, y in zip(ga, gb))
                              / max(1e-12, max(abs(ga[-1] - ga[0]), 1e-12)))
        flag = bool(abs(smd) > smd_flag or (row["ks"] or 0.0) > ks_flag)
        row["flag"] = flag
        out["numeric"].append(row)

    va, vb = fa.get("categorical", {}), fb.get("categorical", {})
    for c in sorted(set(va) & set(vb)):
        da, db = va[c], vb[c]
        tot_a = sum(da.values()) or 1
        tot_b = sum(db.values()) or 1
        keys = set(da) | set(db)
        pa = {k: da.get(k, 0) / tot_a for k in keys}
        pb = {k: db.get(k, 0) / tot_b for k in keys}
        tvd = 0.5 * sum(abs(pa[k] - pb[k]) for k in keys)
        overlap = float(sum(min(pa[k], pb[k]) for k in keys))
        # 取值域几乎不重叠（学号、班级名、换了编码的专业）：这不是「分布漂了」，
        # 而是字段口径变了或本来就是标识列。判成漂移会盖住真正的结论。
        renamed = bool(overlap < 0.10 and tvd > 0.50)
        row = {"column": c, "tvd": float(tvd), "overlap": overlap,
               "flag": bool(tvd > tvd_flag and not renamed),
               "renamed": renamed,
               "new_levels": sorted(set(db) - set(da))[:5],
               "n_new_levels": len(set(db) - set(da)),
               "gone_levels": sorted(set(da) - set(db))[:5],
               "n_gone_levels": len(set(da) - set(db))}
        out["categorical"].append(row)

    out["flagged"] = ([r["column"] for r in out["numeric"] if r["flag"]]
                      + [r["column"] for r in out["categorical"] if r["flag"]])
    out["renamed"] = [r["column"] for r in out["categorical"] if r["renamed"]]
    out["n_flagged"] = len(out["flagged"])
    return out


# ---------------------------------------------------------------- 权重漂移

def compare(a: dict, b: dict, alpha: float = 0.05, schema=None) -> dict:
    """对比两份快照：数据侧漂移 + 权重侧漂移（相对标准误判定）。

    schema：字段映射（Schema / dict / 兼容性结论）。它声明哪些 canonical 字段
    允许跨批次比较；未声明的列即使同名也不参与 Δw 判定，避免拿「专业」
    这学期填专业名、下学期填班级名的列去讲漂移故事。
    """
    blocked: dict[str, str] = {}
    schema_note = None
    budget_note = None
    ba = (a.get("config") or {}).get("speed_budget") or {}
    bb = (b.get("config") or {}).get("speed_budget") or {}
    if ba and bb:
        diff = {k: (ba.get(k), bb.get(k))
                for k in ("pso_particles", "pso_iters", "contribution_rounds",
                          "stab_boot", "subsample_n")
                if ba.get(k) != bb.get(k)}
        if diff:
            budget_note = ("两次分析的算力预算不同（%s）。PSO 是随机搜索，预算小的那次"
                           "解更粗，Δw 里混着搜索精度的差，不只是数据变化。"
                           % "、".join("%s: %s→%s" % (k, v[0], v[1])
                                      for k, v in diff.items()))
    if schema is not None:
        d = schema.get("compatibility", schema) if isinstance(schema, dict) else schema
        if hasattr(d, "to_dict"):
            d = d.to_dict()
        for f in d.get("fields", []):
            if f.get("role") == "feature" and not f.get("comparable", True):
                blocked[str(f.get("name"))] = "字段映射声明该列不可跨批次比较（同名不同义）"
        if isinstance(d, dict) and "not_comparable" in d and not d.get("fields"):
            for n in d.get("not_comparable", []):
                blocked[str(n)] = "字段映射声明该列不可跨批次比较（同名不同义）"
    else:
        fa_s, fb_s = a.get("field_schema"), b.get("field_schema")
        if fa_s and fb_s:
            if (fa_s.get("id"), fa_s.get("version")) != (fb_s.get("id"), fb_s.get("version")):
                schema_note = ("两份快照用的字段映射不同（%s v%s vs %s v%s）："
                               "canonical 字段的含义可能已经变了，对比前先确认映射一致。"
                               % (fa_s.get("id"), fa_s.get("version"),
                                  fb_s.get("id"), fb_s.get("version")))
            for fs in (fa_s, fb_s):
                blocked.update(_blocked_from_schema(fs))

    fa = [str(x) for x in (a.get("feature_names") or [])]
    fb = [str(x) for x in (b.get("feature_names") or [])]
    common = [f for f in fa if f in fb]
    only_a = [f for f in fa if f not in fb]
    only_b = [f for f in fb if f not in fa]

    wa = np.array([a["w"][fa.index(f)] for f in common], float) if common else np.array([])
    wb = np.array([b["w"][fb.index(f)] for f in common], float) if common else np.array([])
    sea = np.array([a["se_w"][fa.index(f)] if a.get("se_w") else np.nan
                    for f in common], float) if common else np.array([])
    seb = np.array([b["se_w"][fb.index(f)] if b.get("se_w") else np.nan
                    for f in common], float) if common else np.array([])

    d = data_drift(a, b)

    testable = np.isfinite(sea) & np.isfinite(seb) & ((sea + seb) > 0)
    # 取值域不重叠的列（同名不同义，比如「专业」一列这学期填专业名、下学期填班级名）
    # 不参与权重对比：比出来的 Δw 没有可解释的含义。
    renamed = set(d.get("renamed") or [])
    for n in renamed:
        blocked.setdefault(n, "该列取值域几乎不重叠（同名不同义），Δw 无可解释含义")
    if renamed:
        testable &= np.array([f not in renamed for f in common], bool)
    if blocked:
        testable &= np.array([f not in blocked for f in common], bool)
    sed = np.where(testable, np.sqrt(np.nan_to_num(sea) ** 2
                                     + np.nan_to_num(seb) ** 2), np.nan)
    dw = wb - wa
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(testable, dw / np.where(sed > 0, sed, 1.0), np.nan)

    p = np.full(common.__len__(), np.nan)
    if testable.any():
        p[testable] = _z_p(z[testable])
    q = np.full(p.size, np.nan)
    sig = np.zeros(p.size, bool)
    n_sig = 0
    if testable.any():
        q_t, sig_t, n_sig = bh_fdr(p[testable], alpha)
        q[testable] = q_t
        sig[testable] = sig_t

    # 整体卡方：只有 m−1 个自由度（权重和为 1）
    chi2 = float(np.nansum(np.where(testable, (dw / np.where(sed > 0, sed, 1.0)) ** 2, 0.0)))
    dfree = max(int(testable.sum()) - 1, 0)
    if dfree > 0 and _st is not None and np.isfinite(chi2):
        p_global = float(_st.chi2.sf(chi2, dfree))
    else:
        p_global = None

    rows = []
    for i, f in enumerate(common):
        if f in blocked:
            verdict = "口径不可比"
            why = blocked[f]
        elif np.isfinite(sea[i]) and np.isfinite(seb[i]) and (sea[i] + seb[i]) <= 0:
            verdict = "不可判"
            why = "标准误为 0：数据近似可分或权重退化，Δw 没有统计意义"
        elif not testable[i]:
            verdict = "不可判"
            why = "缺少标准误（无标签场景，或当时诊断未跑）"
        elif sig[i]:
            verdict = "真漂移"
            why = "Δw 超过其标准误的 %.1f 倍" % abs(z[i])
        else:
            verdict = "只是噪声"
            why = "Δw 未超过标准误（|z|=%.2f）" % abs(z[i])
        rows.append({
            "feature": f, "w_a": float(wa[i]), "w_b": float(wb[i]),
            "delta": float(dw[i]), "se_delta": (float(sed[i])
                                                if np.isfinite(sed[i]) else None),
            "z": (float(z[i]) if np.isfinite(z[i]) else None),
            "p": (float(p[i]) if np.isfinite(p[i]) else None),
            "q": (float(q[i]) if np.isfinite(q[i]) else None),
            "sig": bool(sig[i]), "verdict": verdict, "why": why,
            "direction": ("上升" if dw[i] > 0 else "下降") if dw[i] != 0 else "不变",
        })

    # 三种结论的判定
    if d.get("same_data_pool"):
        kind = "same_data"
        headline = ("两份快照来自同一批样本（数据指纹一致），Δw 只可能来自优化器与"
                    "随机种子，不是真实漂移。")
    elif not testable.any():
        if blocked and set(common) and set(common) <= set(blocked):
            kind = "not_comparable"
            headline = ("共同特征全是被字段映射标为不可比的列（%s），权重无法跨批次比较。"
                        "先统一字段口径，或把映射里这些列的 comparable 打开（确认同义后）。"
                        % "、".join(common))
        else:
            kind = "no_se"
            headline = ("两份快照都没有可用的标准误（无标签场景常见），只能给描述性的 Δw，"
                        "不能判漂移。要判就得给标签列，或在两边都跑 bootstrap。")
    # 三种结论的判定。注意一种组合：整体卡方显著但没有任何单项在 FDR 后站得住
    # （每个 Δw 都不足以单独定案，合起来却超出噪声）——这仍是漂移，只是不能点名。
    elif n_sig > 0:
        kind = "weight_drift"
        headline = "数据变了，权重也显著变了：%d 个特征的 Δw 超过噪声尺度。" % n_sig
    elif p_global is not None and p_global < alpha:
        kind = "weight_drift"
        headline = ("数据变了，权重整体显著漂移（卡方 %.2f，df=%d，p=%.3f），"
                    "但没有单个特征在 FDR 校正后站得住：只能说「变了」，不能点名是谁。"
                    % (chi2, dfree, p_global))
    elif d.get("n_flagged") or d.get("renamed"):
        kind = "data_drift"
        what = d.get("flagged") or []
        tail = ("；另有 %s 的取值域不重叠（口径变化）"
                % "、".join(d["renamed"][:3])) if d.get("renamed") else ""
        headline = ("数据侧已经变了（%s%s），但权重差异还在噪声范围内：结论是稳的。"
                    % ("、".join(what[:5]) if what else "字段口径变化", tail))
    else:
        kind = "stable"
        headline = "数据与权重都没检出漂移：两次分析可视为复现。"

    caveats: list[str] = []
    if blocked:
        shift = [(abs(dw[i]), f) for i, f in enumerate(common) if f in blocked]
        if shift:
            top = max(shift)
            if top[0] > 0.05:
                caveats.append(
                    "被排除的列「%s」自身权重差 %.3f。权重和为 1，它多占/让出的份额必然"
                    "分摊到其余列上：其余列的 Δw 里混着这个份额效应，不能直接读成"
                    "「该项的作用变了」。要干净地比，就用只含可比字段的模型再跑一次。"
                    % (top[1], top[0]))

    out = {
        "kind": kind,
        "headline": headline,
        "alpha": float(alpha),
        "blocked": {k: v for k, v in blocked.items()},
        "caveats": caveats,
        "budget_note": budget_note,
        "schema_note": schema_note,
        "a": {"name": a.get("name"), "source": a.get("source"),
              "created_at": a.get("created_at"), "n": a.get("n"),
              "version": a.get("version")},
        "b": {"name": b.get("name"), "source": b.get("source"),
              "created_at": b.get("created_at"), "n": b.get("n"),
              "version": b.get("version")},
        "data": d,
        "features_common": common,
        "features_only_in_a": only_a,
        "features_only_in_b": only_b,
        "rows": rows,
        "n_testable": int(testable.sum()),
        "n_sig": int(n_sig),
        "l1": float(np.nansum(np.abs(dw))),
        "l2": float(np.sqrt(np.nansum(dw ** 2))),
        "chi2": chi2 if np.isfinite(chi2) else None,
        "chi2_df": dfree,
        "p_global": p_global,
        "cluster": {
            "n_clusters": {"a": a.get("n_clusters"), "b": b.get("n_clusters")},
            "sizes": {"a": a.get("cluster_sizes"), "b": b.get("cluster_sizes")},
            "stability": {"a": (a.get("stability") or {}).get("level"),
                          "b": (b.get("stability") or {}).get("level")},
        },
    }
    out["verdict_text"] = to_text(out)
    return out


# ---------------------------------------------------------------- 可读文本

def to_text(rep: dict) -> str:
    L = ["跨批次对比：%s → %s"
         % (rep["a"].get("name") or rep["a"].get("source") or "A",
            rep["b"].get("name") or rep["b"].get("source") or "B")]
    L.append("  样本 %s → %s，共同特征 %d 个"
             % (rep["a"].get("n"), rep["b"].get("n"), len(rep["features_common"])))
    if rep.get("features_only_in_a") or rep.get("features_only_in_b"):
        L.append("  [注意] 特征不一致：仅 A 有 %s；仅 B 有 %s"
                 % ("、".join(rep["features_only_in_a"]) or "无",
                    "、".join(rep["features_only_in_b"]) or "无"))
    if rep.get("schema_note"):
        L.append("  [字段映射] %s" % rep["schema_note"])
    if rep.get("budget_note"):
        L.append("  [算力预算] %s" % rep["budget_note"])
    blocked = rep.get("blocked") or {}
    common = rep.get("features_common") or []
    hit = sorted(set(blocked) & set(common))
    if hit:
        L.append("  [字段映射] 以下列不参与 Δw 判定：%s（%s）"
                 % ("、".join(hit), "；".join(blocked[n] for n in hit)))

    d = rep["data"]
    L.append("")
    L.append("  数据侧：%s" % ("指纹一致（同一批样本，行序%s）"
                            % ("相同" if d.get("same_data_rows") else "不同")
                            if d.get("same_data_pool")
                            else ("指纹不同" if d.get("has_fingerprint")
                                  else "缺指纹，无法核对")))
    if d.get("note"):
        L.append("    [提示] %s" % d["note"])
    if d.get("columns_only_in_a") or d.get("columns_only_in_b"):
        L.append("    字段差异：仅 A 有 %s；仅 B 有 %s"
                 % ("、".join(d["columns_only_in_a"]) or "无",
                    "、".join(d["columns_only_in_b"]) or "无"))
    for r in d.get("numeric", []):
        flag = " ← 漂" if r["flag"] else ""
        L.append("    数值 %-14s 均值 %.3f → %.3f（Δ%+.3f，SMD=%+.2f）%s"
                 % (r["column"], r["mean_a"], r["mean_b"], r["delta"], r["smd"], flag))
    for r in d.get("categorical", []):
        if r.get("renamed"):
            mark = " ← 取值域几乎不重叠（口径变了 / 疑似标识列），不计为漂移"
        else:
            mark = " ← 漂" if r["flag"] else ""
        tail = ""
        if r.get("n_new_levels"):
            tail += " 新档 %d 个" % r["n_new_levels"]
        if r.get("n_gone_levels"):
            tail += " 消失档 %d 个" % r["n_gone_levels"]
        L.append("    类别 %-14s 分布距离 %.3f（取值域重合 %.2f）%s%s"
                 % (r["column"], r["tvd"], r.get("overlap", 0.0), mark, tail))
    if d.get("renamed"):
        L.append("    取值域不重叠的列：%s —— 这些列的同名不代表同义，"
                 "权重对比时不要跨批次解释它们。" % "、".join(d["renamed"]))

    L.append("")
    L.append("  权重侧：L1=%.3f  L2=%.3f  可检验特征 %d 个，显著 %d 个"
             % (rep["l1"], rep["l2"], rep["n_testable"], rep["n_sig"]))
    if rep.get("chi2") is not None and rep.get("p_global") is not None:
        L.append("    整体卡方 %.2f（df=%d），p=%.3f%s"
                 % (rep["chi2"], rep["chi2_df"], rep["p_global"],
                    "" if rep["p_global"] >= rep["alpha"] else " → 整体漂移显著"))
    L.append("    %-14s %8s %8s %8s %8s  %s"
             % ("feature", "w_A", "w_B", "Δw", "z", "判定"))
    for r in rep["rows"]:
        L.append("    %-14s %8.4f %8.4f %+8.4f %8s  %s"
                 % (r["feature"], r["w_a"], r["w_b"], r["delta"],
                    "%.2f" % r["z"] if r["z"] is not None else "n/a",
                    r["verdict"]))

    L.append("")
    L.append("  结论：" + rep["headline"])
    for c in rep.get("caveats") or []:
        L.append("    [注意] %s" % c)
    if rep["kind"] == "same_data":
        L.append("    这种情况下不要把 Δw 写成「趋势」。要估优化器噪声，就固定数据、"
                 "换种子重跑若干次，把 Δw 的分布当作基线。")
    elif rep["kind"] == "no_se":
        L.append("    建议：给标签列重跑（拿到 sandwich 标准误），或两边都开 bootstrap。")
    elif rep["kind"] == "weight_drift":
        for r in rep["rows"]:
            if r["sig"]:
                L.append("    · %s：%.4f → %.4f（%s %+.4f，%s，q=%.3f）"
                         % (r["feature"], r["w_a"], r["w_b"], r["direction"],
                            r["delta"], r["why"], r["q"]))
    return "\n".join(L)
