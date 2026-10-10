# -*- coding: utf-8 -*-
"""学情分型案例 v2：数据体检 + τ̂ 判据 + 模板解读，跑两个学期的真实成绩。

用法：python cases/learning_analytics/run_case.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hybridscope.estimator import HybridScopeEstimator
from hybridscope.io import split_columns
from hybridscope.prep import Prep
from hybridscope.schema import Schema, compat_text
from hybridscope.templates import grades as gr

DATA = Path(__file__).resolve().parent / "data"
TARGET_BAND = "学情档位"


def run_one(path: Path, title: str, k: int = 5):
    df = gr.load(path)
    roles = gr.detect_roles(df)
    target = gr.pick_target(df, roles)
    num, cat = gr.feature_split(df, target=target, roles=roles)
    summ = gr.summarize(df, target=target, roles=roles)

    print("=" * 72)
    print(title)
    print("=" * 72)
    print("学生 %d 人，考核项 %d 个：%s" % (summ["n_students"], len(num), "、".join(num)))
    if summ["id_columns"]:
        print("标识列（不进特征）：%s" % "、".join(summ["id_columns"]))

    df, rep = gr.drop_absent_rows(df, target=target, roles=roles)
    if rep["dropped_absent"]:
        print("剔除缺考/未参与 %d 人，余 %d 人" % (rep["dropped_absent"], rep["kept"]))

    work = df.copy()
    work[TARGET_BAND] = gr.band(work[target])
    print("学情档位分布：%s"
          % work[TARGET_BAND].value_counts().reindex(list(gr.BAND_NAMES)).to_dict())

    sub = work[num + cat + [TARGET_BAND]]
    prep = Prep()
    clean = prep.fit_transform(sub, target=TARGET_BAND, cat_cols=cat)
    print("\n" + prep.to_text())

    X_num, X_cat, num_names, cat_names, y = split_columns(
        clean, target=TARGET_BAND, cat_cols=cat)
    est = HybridScopeEstimator(k=min(k, len(clean) - 1), seed=0, auto_prep=False)
    est.fit_hybrid(X_num, X_cat, y, feature_names=num_names + cat_names)
    s = est.summary()

    a = est.advisor_
    print("\n[去噪顾问] %s（%s）" % ("启用" if a["denoise"] else "关闭", a["rule"]))
    print("           acf(1)=%.3f  τ̂=%s（%s）"
          % (a["acf1_median"],
             "%.3f" % a["tau_median"] if np.isfinite(a["tau_median"]) else "n/a",
             a["tau_band"]))
    if a.get("sort_key"):
        sk = a["sort_key"]
        print("           排序键检验：相邻同组 %.0f%%，随机期望 %.0f%%（%.1f×）→ %s"
              % (sk["same_adj"] * 100, sk["expected"] * 100, sk["ratio"],
                 "行序由分组键决定" if sk["blocked"] else "未见分块存放"))
    print("           %s" % a["reason"])
    if a.get("caveat"):
        print("           [提示] %s" % a["caveat"])

    print("[分型质量] ACC=%.4f  NMI=%.4f  ARI=%.4f"
          % (s["metrics"]["ACC"], s["metrics"]["NMI"], s["metrics"]["ARI"]))

    dg = est.diagnostics_
    if not dg or "rows" not in dg:
        print("\n无诊断结果")
        return est, df
    print("[诊断] 可分=%s  病态=%s  cond_H=%.3g  rho_trace=%.3g"
          % (dg["separable"], dg["unstable"], dg["cond_H"], dg["rho_trace"]))
    if dg.get("identifiable") is False:
        print("  权重不可识别：%s（所有结论按红灯处理）" % dg.get("unstable_reason"))
    if dg["separable"]:
        print("        注意：数据近似可分，权重不可辨识，诊断仅供参考。")
    fdr = s.get("fdr") or {}
    if fdr:
        print("        多重检验校正：BH-FDR α=%.2f，%d 个权重同时检验，显著 %d 个"
              % (fdr["alpha"], fdr["n_tests"], fdr["n_sig"]))
    print("\n%-10s %8s %8s %8s %9s %5s  %-12s %s"
          % ("特征", "权重", "标准误", "z", "q", "显著", "判定", "达标样本"))
    for r in dg["rows"]:
        print("%-10s %8.4f %8.4f %8.2f %9.2e %5s  %-12s %s"
              % (r["feature"], r["weight"], r["se"], r["z"], r.get("q", float("nan")),
                 "是" if r.get("sig") else "-", r["level_zh"],
                 str(r["n_needed"]) if r["reachable"] else "不可达"))

    ok = [r["feature"] for r in dg["rows"] if r["level_fdr"] == "green"]
    print("\n校正后仍显著（可信）的项：%s" % ("、".join(ok) if ok else "无"))
    print("解读：")
    for e in gr.explain(dg["rows"], cat_names=cat_names):
        print("  · " + e["text"])

    if getattr(est, "stability_", None) and "level" in est.stability_:
        from hybridscope.stability import to_text
        print("\n" + to_text(est.stability_))
    return est, df


def _print_fit(est, cat_names):
    """打印一次拟合的顾问/质量/诊断/解读，两个层级共用。"""
    a = est.advisor_
    print("\n[去噪顾问] %s（%s）" % ("启用" if a["denoise"] else "关闭", a["rule"]))
    print("           acf(1)=%.3f  τ̂=%s（%s）"
          % (a["acf1_median"],
             "%.3f" % a["tau_median"] if np.isfinite(a["tau_median"]) else "n/a",
             a["tau_band"]))
    if a.get("sort_key"):
        sk = a["sort_key"]
        print("           排序键检验：相邻同组 %.0f%%，随机期望 %.0f%%（%.1f×）→ %s"
              % (sk["same_adj"] * 100, sk["expected"] * 100, sk["ratio"],
                 "行序由分组键决定" if sk["blocked"] else "未见分块存放"))
    print("           %s" % a["reason"])
    if a.get("caveat"):
        print("           [提示] %s" % a["caveat"])

    s = est.summary()
    print("[分型质量] ACC=%.4f  NMI=%.4f  ARI=%.4f"
          % (s["metrics"]["ACC"], s["metrics"]["NMI"], s["metrics"]["ARI"]))

    dg = est.diagnostics_
    if not dg or "rows" not in dg:
        print("\n无诊断结果")
        return
    print("[诊断] 可分=%s  病态=%s  cond_H=%.3g  rho_trace=%.3g"
          % (dg["separable"], dg["unstable"], dg["cond_H"], dg["rho_trace"]))
    if dg.get("identifiable") is False:
        print("  权重不可识别：%s（所有结论按红灯处理）" % dg.get("unstable_reason"))
    if dg["separable"]:
        print("        注意：数据近似可分，权重不可辨识，诊断仅供参考。")
    fdr = s.get("fdr") or {}
    if fdr:
        print("        多重检验校正：BH-FDR α=%.2f，%d 个权重同时检验，显著 %d 个"
              % (fdr["alpha"], fdr["n_tests"], fdr["n_sig"]))
    print("\n%-10s %8s %8s %8s %9s %5s  %-12s %s"
          % ("特征", "权重", "标准误", "z", "q", "显著", "判定", "达标样本"))
    for r in dg["rows"]:
        print("%-10s %8.4f %8.4f %8.2f %9.2e %5s  %-12s %s"
              % (r["feature"], r["weight"], r["se"], r["z"], r.get("q", float("nan")),
                 "是" if r.get("sig") else "-", r["level_zh"],
                 str(r["n_needed"]) if r["reachable"] else "不可达"))

    ok = [r["feature"] for r in dg["rows"] if r["level_fdr"] == "green"]
    print("\n校正后仍显著（可信）的项：%s" % ("、".join(ok) if ok else "无"))
    print("解读：")
    for e in gr.explain(dg["rows"], cat_names=cat_names):
        print("  · " + e["text"])

    if getattr(est, "stability_", None) and "level" in est.stability_:
        from hybridscope.stability import to_text
        print("\n" + to_text(est.stability_))


def run_unified(path: Path, schema, out: Path, k: int = 5):
    """统一口径层：两学期用同一套 canonical 字段建模，结论才跨学期可比。

    每个学期跑两个模型：
      · 可比模型（过程考核 + 项目考核）：跨学期权重对比只用这一个。
        不带分组列，是因为两学期的分组一个是专业、一个是行政班，语义不同；
        带上它，它的权重会从 0 变到 0.64，被挤占的份额会落到其余列头上，
        Δw 就混进了「分组维度有没有信息」，而不是「考核项作用变没变」。
      · 完整模型（再加分组）：用于学期内解读，不用于跨学期比较。
    """
    from hybridscope import monitor
    from hybridscope.report import write_report

    long = pd.read_csv(path, encoding="utf-8-sig")
    num, cat = ["过程考核", "项目考核"], ["分组"]
    aliases = {"process": "过程考核", "project": "项目考核", "cohort": "分组"}
    snaps = {}
    for term in sorted(long["学期"].unique()):
        g = long[long["学期"] == term].reset_index(drop=True)
        print("\n" + "=" * 72)
        print("%s：学期内完整口径（%d 人，特征：%s + %s）"
              % (term, len(g), "、".join(num), "、".join(cat)))
        print("=" * 72)
        sub = g[num + cat + ["学情档位"]].copy()
        prep = Prep()
        clean = prep.fit_transform(sub, target="学情档位", cat_cols=cat)
        print(prep.to_text())
        for cols, tag in ((num + cat, "完整"), (num, "可比")):
            ccat = cat if tag == "完整" else []
            est = HybridScopeEstimator(k=min(k, len(clean) - 1), seed=0, auto_prep=False)
            if tag == "可比":
                # 只把可比字段喂进去：连列都不出现，才不会在单纯形上抢份额
                sub_c = clean[num + ["学情档位"]]
                X_num, X_cat, num_names, cat_names, y = split_columns(
                    sub_c, target="学情档位", cat_cols=[])
                est.fit_hybrid(X_num, X_cat, y, feature_names=list(num_names))
            else:
                X_num, X_cat, num_names, cat_names, y = split_columns(
                    clean, target="学情档位", cat_cols=ccat)
                est.fit_hybrid(X_num, X_cat, y, feature_names=num_names + cat_names)
            print("\n--- %s口径模型（%s）---"
                  % (tag, "、".join(list(num_names) + list(cat_names))))
            _print_fit(est, cat_names=cat_names)
            if tag == "可比":
                snaps[term] = est.snapshot(
                    df=g[["过程考核", "项目考核", "学情档位"]], name=term,
                    source=path.name, schema=schema, profile=term, aliases=aliases,
                    notes="统一口径·仅可比字段（canonical process/project）")
                monitor.save_snapshot(snaps[term],
                                      str(out / ("snap_unified_%s.json" % term)))
                write_report(str(out / ("report_unified_%s.html" % term)), est,
                             df=g[["过程考核", "项目考核", "学情档位"]],
                             source=path.name, template=gr,
                             title="HybridScope 学情分型报告（统一口径 · %s）" % term)
                print("\n已写出 snap_unified_%s.json / report_unified_%s.html"
                      % (term, term))

    terms = sorted(snaps)
    rep = monitor.compare(snaps[terms[0]], snaps[terms[1]])
    print("\n" + "=" * 72)
    print("统一口径下的跨学期对比（只用可比字段：过程考核、项目考核）")
    print("=" * 72)
    print(rep["verdict_text"])
    return snaps, rep


def main():
    from hybridscope import monitor
    from hybridscope.report import write_report

    out = Path(__file__).resolve().parent
    e_spring, df_spring = run_one(
        DATA / "grades_2026spring.csv",
        "案例 A · 2026 春《人工智能导论》(100514T003)")
    print("\n")
    e_autumn, df_autumn = run_one(
        DATA / "grades_2025autumn.csv",
        "案例 B · 2025 秋《人工智能通识（信息类）》(102000C001)")

    # 存档 + 报告：一次分析的产物要能被下次核对
    snaps = {}
    for tag, est, frame in (("2026spring", e_spring, df_spring),
                            ("2025autumn", e_autumn, df_autumn)):
        snaps[tag] = est.snapshot(df=frame, name=tag, source="grades_%s.csv" % tag)
        monitor.save_snapshot(snaps[tag], str(out / ("snap_%s.json" % tag)))
        write_report(str(out / ("report_%s.html" % tag)), est, df=frame,
                     source="grades_%s.csv" % tag, title="HybridScope 学情分型报告（%s）"
                     % tag, template=gr)
        print("\n已写出 snap_%s.json / report_%s.html" % (tag, tag))

    print("\n" + "=" * 72)
    print("跨学期对比：权重变了是真漂移，还是只是噪声？")
    print("=" * 72)
    rep = monitor.compare(snaps["2026spring"], snaps["2025autumn"])
    print(rep["verdict_text"])
    print("\n  上面这一层是各学期自己的分项（春 8 项 / 秋 2 项），本来就不该直接比；")
    print("  要谈跨学期漂移，得先把字段口径对齐，见下一节。")

    # ---- 统一口径层 ----
    frames = {p: gr.load(DATA / ("grades_%s.csv" % p))
              for p in ("2026spring", "2025autumn")}
    schema = gr.canonical_schema()
    sp = out / "schema_grades.json"
    if sp.exists():
        schema = Schema.from_json(str(sp))
    print("\n" + "=" * 72)
    print("统一字段口径：两学期 → canonical 字段")
    print("=" * 72)
    print(compat_text(schema.compatibility(frames)))
    for c in schema.compatibility(frames).get("cautions", []):
        print("  [提示] %s" % c)

    print("\n" + "=" * 72)
    print("统一口径层：同一套字段跑两学期，再比权重")
    print("=" * 72)
    run_unified(DATA / "grades_unified.csv", schema, out)


if __name__ == "__main__":
    main()
