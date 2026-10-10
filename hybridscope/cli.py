# -*- coding: utf-8 -*-
"""命令行入口：python -m hybridscope.cli data.csv --target y

新增（v2）：
  --prep auto|on|off      数据体检与修补（默认 auto，即开）
  --template grades       学情分型模板：自动认字段、翻译权重
  --prep-json FILE        把体检报告单独存一份

新增（v2.1）：
  --fdr-alpha 0.05        多权重同时检验的 BH-FDR 显著性水平
  --boot 200              sandwich 的梯度 bootstrap 次数（0 关闭）
  --stab-boot/--stab-perm 无标签可信度的 bootstrap / 置换次数
  --no-stability          关掉稳定性检验（大表提速）
新增（v2.2）：
  --report FILE.html     一键 HTML 报告（含数据指纹、复现参数、动作清单）
  --snapshot FILE.json   把本次结果存成快照，供以后对比
  --compare FILE.json    与旧快照对比：区分「真漂移 / 只是噪声 / 数据先变了」

新增（v2.3）：
  --schema FILE.json     按字段映射把原始列折叠成 canonical 字段后再分析
  --profile NAME         该批次在映射里的名字（省略则用文件名主干匹配）
  --map-only             只做映射与体检，不建模
  --unify OUT.csv        多张表按同一映射对齐，输出长表 + 可比性结论
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from hybridscope.estimator import HybridScopeEstimator
from hybridscope.io import split_columns
from hybridscope.prep import Prep


def read_any(path: str) -> pd.DataFrame:
    return pd.read_csv(path) if str(path).lower().endswith(".csv") else pd.read_excel(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hybridscope",
                                 description="混合属性分型与权重可估计性诊断")
    ap.add_argument("input", nargs="+", help="CSV 或 Excel 文件（--unify 时可给多个）")
    ap.add_argument("--target", default=None, help="标签列名；省略则为无监督聚类")
    ap.add_argument("--cat", default=None,
                    help="逗号分隔的类别列名（整数编码的类别列需手动指定）")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--denoise", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--particles", type=int, default=15)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-diagnose", action="store_true")
    ap.add_argument("--json", default=None, help="把结果写成 JSON")
    ap.add_argument("--prep", default="auto", choices=["auto", "on", "off"],
                    help="数据体检与修补（缺失/离群/低频档/标识列/量纲）")
    ap.add_argument("--template", default=None, choices=["grades", "survey", "ops"],
                    help="领域模板：grades 学情分型 / survey 问卷量表 / ops 设备运维")
    ap.add_argument("--prep-json", default=None, help="把数据体检报告写成 JSON")
    ap.add_argument("--fdr-alpha", type=float, default=0.05,
                    help="多权重同时检验的 BH-FDR 水平（默认 0.05）")
    ap.add_argument("--boot", type=int, default=200,
                    help="sandwich 的梯度 bootstrap 次数，0 关闭（默认 200）")
    ap.add_argument("--stab-boot", type=int, default=60,
                    help="稳定性检验的 bootstrap 次数（默认 60）")
    ap.add_argument("--stab-perm", type=int, default=60,
                    help="稳定性检验的置换次数（默认 60）")
    ap.add_argument("--no-stability", action="store_true",
                    help="关掉稳定性检验（大表提速）")
    ap.add_argument("--report", default=None, help="生成 HTML 报告（自包含，可转发）")
    ap.add_argument("--snapshot", default=None, help="把本次结果存成 JSON 快照")
    ap.add_argument("--compare", default=None,
                    help="与旧快照对比，判断权重是真漂移还是噪声")
    ap.add_argument("--name", default=None, help="本次分析的批次名（如 2026春）")
    ap.add_argument("--schema", default=None,
                    help="字段映射 JSON：把各批次原始列折叠到 canonical 字段")
    ap.add_argument("--profile", default=None,
                    help="该批次在映射里的名字（省略则用文件名主干匹配）")
    ap.add_argument("--map-only", action="store_true",
                    help="只做映射与审计，不建模")
    ap.add_argument("--unify", default=None,
                    help="多张表按同一映射对齐后写出长表 CSV（不建模）")
    ap.add_argument("--speed", default="off", choices=["off", "auto", "fast", "thorough"],
                    help="算力预算：auto 随样本量降档，fast 先求跑完，thorough 不压")
    ap.add_argument("--max-n", type=int, default=None,
                    help="稳定性检验的子样本上限（大表降精度换时间）")
    ap.add_argument("--drop-invalid", action="store_true",
                    help="问卷模板：剔除直线作答（所有题选同一选项）的问卷")
    ap.add_argument("--dims", default=None,
                    help="问卷模板：维度定义 JSON，{\"维度名\": [\"Q1\",\"Q2\"]}")
    ap.add_argument("--dry-run", action="store_true",
                    help="只估算耗时与内存并给建议，不建模")
    args = ap.parse_args(argv)

    # ---- 多表对齐：先折叠再合并，跨批次可比性在这里定 ----
    if args.unify:
        from hybridscope.schema import Schema, audit_text, compat_text, harmonize
        sch = Schema.from_json(args.schema) if args.schema else None
        if sch is None:
            print("[错误] --unify 需要 --schema 指定字段映射 JSON")
            return 2
        frames = {}
        for p in args.input:
            prof = args.profile or Path(p).stem
            if prof not in sch.profiles:
                cand = [k for k in sch.profiles if k in str(p)]
                prof = cand[0] if cand else prof
            if prof not in sch.profiles:
                print("[错误] %s 在映射里找不到对应批次（已有：%s）"
                      % (p, "、".join(sch.profiles)))
                return 2
            frames[prof] = read_any(p)
        for prof, fr in frames.items():
            print(audit_text(sch.apply(fr, prof)))
            print()
        print(compat_text(sch.compatibility(frames)))
        long, _ = harmonize(sch, frames)
        long.to_csv(args.unify, index=False, encoding="utf-8-sig")
        print("\n已写出 %s（%d 行 × %d 列）" % (args.unify, len(long), long.shape[1]))
        return 0

    path = args.input[0]
    df = read_any(path)
    tmpl = None
    cat_cols = [c.strip() for c in args.cat.split(",") if c.strip()] if args.cat else None
    target = args.target
    schema = None
    profile = None

    # ---- 单表按映射折叠：建模列换成 canonical 字段，后续流程不变 ----
    if args.schema:
        from hybridscope.schema import Schema, audit_text
        schema = Schema.from_json(args.schema)
        profile = args.profile or Path(path).stem
        if profile not in schema.profiles:
            cand = [k for k in schema.profiles if k in str(path)]
            profile = cand[0] if cand else profile
        if profile not in schema.profiles:
            print("[错误] 映射里没有批次 %r（已有：%s）"
                  % (profile, "、".join(schema.profiles)))
            return 2
        uni = schema.apply(df, profile)
        print(audit_text(uni))
        if not uni.ok():
            print("[错误] 字段映射未通过：%s" % "；".join(uni.report["errors"]))
            return 2
        df = uni.df
        feat = [f.name for f in schema.fields
                if f.role == "feature" and f.name in df.columns]
        cat_cols = [f.name for f in schema.fields
                    if f.role == "feature" and f.kind == "categorical"
                    and f.name in df.columns]
        if target is None:
            tg = [f.name for f in schema.fields if f.role == "target"
                  and f.name in df.columns]
            target = tg[0] if tg else None
        # 总评是连续分，不能直接当标签；用模板的档位切法转成离散档位，
        # 并把总评本身剔除出特征（否则标签泄漏，权重会假性显著）。
        if target and args.template == "grades":
            from hybridscope.templates import grades as _gr
            lab = "学情档位"
            df[lab] = _gr.band(pd.to_numeric(df[target], errors="coerce"))
            target = lab
        keep = [c for c in feat + ([target] if target else []) if c in df.columns]
        df = df[keep]
        print("[映射] 批次 %s → canonical 字段：%s（类别：%s，标签：%s）"
              % (profile, "、".join(feat), "、".join(cat_cols) or "无", target or "无"))
        if args.map_only:
            out = args.json or (str(path).rsplit(".", 1)[0] + "_canonical.csv")
            df.to_csv(out, index=False, encoding="utf-8-sig")
            print("已写出 %s（%d 行 × %d 列）" % (out, len(df), df.shape[1]))
            return 0

    if args.template == "grades":
        from hybridscope.templates import grades as tmpl
        roles = tmpl.detect_roles(df)
        if target is None:
            target = tmpl.pick_target(df, roles)
            if target:
                print("[模板] 自动选定标签列：%s" % target)
        tnum, tcat = tmpl.feature_split(df, target=target, roles=roles)
        if cat_cols is None:
            cat_cols = tcat
        summ = tmpl.summarize(df, target=target, roles=roles)
        print("[模板] 学生 %d 人，考核项 %d 个：%s"
              % (summ["n_students"], len(tnum), "、".join(tnum)))
        if summ["id_columns"]:
            print("[模板] 标识列（不进特征）：%s" % "、".join(summ["id_columns"]))
        miss = {k: v for k, v in summ["absent_counts"].items() if v}
        if miss:
            print("[模板] 缺考/缺项人数：%s"
                  % "、".join("%s %d 人" % (k, v) for k, v in miss.items()))
        df, rep = tmpl.drop_absent_rows(df, target=target, roles=roles)
        if rep["dropped_absent"]:
            print("[模板] 剔除缺考/未参与 %d 人，余 %d 人"
                  % (rep["dropped_absent"], rep["kept"]))

    elif args.template in ("survey", "ops"):
        from hybridscope.templates import get as _get_tmpl
        tmpl = _get_tmpl(args.template)
        roles = tmpl.detect_roles(df)
        if target is None:
            target = tmpl.pick_target(df, roles)
            if target:
                print("[模板] 自动选定标签列：%s" % target)
        tnum, tcat = tmpl.feature_split(df, target=target, roles=roles)
        if cat_cols is None:
            cat_cols = tcat
        summ = tmpl.summarize(df, target=target, roles=roles)
        if args.template == "survey":
            print("[模板] 问卷 %d 份，题目 %d 个"
                  % (summ["n_respondents"], summ["n_items"]))
            dropped = 0
            if args.drop_invalid:
                df, dropped = tmpl.drop_invalid(df, tnum)
                if dropped:
                    print("[模板] 剔除直线作答 %d 份，余 %d 份"
                          % (dropped, len(df)))
            if args.dims:
                # 维度合成：先按维度合并再建模，别把 20 道题当 20 个特征
                import json as _json
                raw = args.dims
                if Path(raw).exists():
                    raw = Path(raw).read_text(encoding="utf-8")
                cfg = _json.loads(raw)
                reverse = cfg.pop("_reverse", [])
                scale = tuple(cfg.pop("_scale", [1, 5]))
                dims = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
                scores, audit = tmpl.dimension_scores(df, dims, reverse=reverse,
                                                      scale=scale)
                print("[模板] 维度合成：")
                for dname, info in audit["dimensions"].items():
                    a = info.get("alpha")
                    print("        %-12s %2d 题  α=%-6s 均值 %.2f%s"
                          % (dname, info["n_items"],
                             "%.2f" % a if a is not None else "n/a",
                             info["mean"] or 0.0,
                             "  ← 一致性偏低" if info.get("low_alpha") else ""))
                for w in audit["warnings"]:
                    print("[模板] 注意：%s" % w)
                rest = [c for c in (cat_cols or []) if c in df.columns]
                if target and target in df.columns:
                    rest.append(target)
                df = pd.concat([scores, df[rest]], axis=1)
                tnum = list(scores.columns)
        else:
            print("[模板] 采样 %d 条，通道 %d 个：%s"
                  % (summ["n_rows"], len(tnum), "、".join(tnum[:8])
                     + ("…" if len(tnum) > 8 else "")))
            bal = summ.get("balance") or {}
            if bal.get("n_classes"):
                print("[模板] 状态分布：%s"
                      % "、".join("%s %.0f%%" % (k, v * 100)
                                  for k, v in bal["share"].items()))
        if summ.get("id_columns"):
            print("[模板] 标识列（不进特征）：%s" % "、".join(summ["id_columns"]))
        for c in tmpl.caveats(summ):
            print("[模板] 注意：%s" % c)

    # 指纹按体检前的表算：体检是修补动作，不该混进「是不是同一批数据」的判断
    df_raw = df.copy()

    prep = None
    if args.prep in ("auto", "on"):
        prep = Prep()
        df = prep.fit_transform(df, target=target, cat_cols=cat_cols)
        print("\n" + prep.to_text())
        if args.prep_json:
            with open(args.prep_json, "w", encoding="utf-8") as f:
                json.dump(prep.report(), f, ensure_ascii=False, indent=2)
            print("体检报告已写出 %s" % args.prep_json)
    elif args.prep == "off":
        print("[体检] 已关闭，沿用整行丢弃缺失的旧行为。")

    X_num, X_cat, num_names, cat_names, y = split_columns(
        df, target=target, cat_cols=cat_cols)
    if y is None:
        print("[info] 未指定 --target：走无监督路线，以轮廓系数拟合权重，"
              "并用 bootstrap 聚类稳定性 + 置换零分布给可信度。")

    if args.dry_run:
        from hybridscope.speed import advice, budget
        n = len(df)
        m = max(1, sum(1 for c in df.columns
                       if c != target and pd.api.types.is_numeric_dtype(df[c])))
        print(advice(n, m, d_cat=len(cat_cols or []),
                     particles=args.particles, iters=args.iters,
                     stab_boot=args.stab_boot, stab_perm=args.stab_perm,
                     n_boot=args.boot,
                     run_stability=not args.no_stability))
        for mode in ("auto", "fast"):
            b = budget(n, m, len(cat_cols or []), mode, not args.no_stability)
            print("  %-5s 预算：PSO %d×%d，稳定性 %d/%d，子样本 %s"
                  % (mode, b["pso_particles"], b["pso_iters"], b["stab_boot"],
                     b["stab_perm"], b["subsample_n"] or "全样本"))
        return 0

    keep = []
    raw = list(argv) if argv is not None else list(sys.argv[1:])
    for key, flag in (("pso_particles", "--particles"), ("pso_iters", "--iters"),
                      ("n_boot", "--boot"), ("stab_boot", "--stab-boot"),
                      ("stab_perm", "--stab-perm")):
        if flag in raw:
            keep.append(key)
    denoise = {"auto": "auto", "on": True, "off": False}[args.denoise]
    est = HybridScopeEstimator(k=args.k, denoise=denoise,
                               pso_particles=args.particles,
                               pso_iters=args.iters,
                               run_diagnose=not args.no_diagnose,
                               fdr_alpha=args.fdr_alpha,
                               n_boot=args.boot,
                               run_stability=not args.no_stability,
                               stab_boot=args.stab_boot,
                               stab_perm=args.stab_perm,
                               speed=args.speed, max_n=args.max_n,
                               speed_keep=tuple(keep),
                               auto_prep=False)
    est.fit_hybrid(X_num, X_cat, y, feature_names=num_names + cat_names)

    s = est.summary()
    print("\n样本 %d  特征 %d（数值 %d / 类别 %d）  k=%d"
          % (s["n"], len(num_names) + len(cat_names), len(num_names), len(cat_names), s["k"]))
    if est.advisor_ is not None:
        a = est.advisor_
        print("去噪顾问: %s —— %s" % (a["denoise"], a["reason"]))
        if a.get("tau_median") is not None:
            import math
            tau = a["tau_median"]
            print("          acf(1)=%s  τ̂=%s（%s）"
                  % ("%.3f" % a["acf1_median"] if a.get("acf1_median") == a.get("acf1_median") else "-",
                     ("%.3f" % tau) if not math.isnan(tau) else "n/a", a.get("tau_band")))
        if a.get("caveat"):
            print("          [提示] %s" % a["caveat"])
    if s["metrics"]:
        print("指标: " + "  ".join("%s=%.4f" % (k, v) for k, v in s["metrics"].items()))
    if s.get("speed"):
        sp = s["speed"]
        print("算力预算: %s（PSO %d×%d，稳定性 %d/%d，bootstrap %d，子样本 %s%s）"
              % (sp.get("mode"), sp.get("pso_particles"), sp.get("pso_iters"),
                 sp.get("stab_boot"), sp.get("stab_perm"), sp.get("n_boot"),
                 sp.get("subsample_n") or "全样本",
                 "，低内存模式" if sp.get("low_memory") else ""))
        if sp.get("note"):
            print("          %s" % sp["note"])

    if s.get("diagnostics"):
        tmap = ({f.name: f.label for f in schema.fields if f.label}
                if schema is not None else None)
        if s.get("separable"):
            print("[warn] 数据近似线性可分，权重不可辨识，诊断仅供参考。")
        if s.get("identifiable") is False:
            print("[warn] 权重不可识别：%s 所有结论按红灯处理。"
                  % (s.get("unstable_reason") or "数值不稳定"))
        elif s.get("unstable"):
            print("[warn] H 病态（cond=%.3g），诊断数值不稳定。" % s["cond_H"])
        fdr = s.get("fdr") or {}
        print("多重检验校正：BH-FDR α=%.2f，%d 个权重同时检验，显著 %d 个"
              % (fdr.get("alpha", args.fdr_alpha), fdr.get("n_tests", 0),
                 fdr.get("n_sig", 0)))
        print("%-16s %8s %8s %8s %9s %6s  %-12s %s"
              % ("feature", "weight", "se", "z", "q", "sig", "verdict", "n_needed"))
        for r in s["diagnostics"]:
            print("%-16s %8.4f %8.4f %8.2f %9.2e %6s  %-12s %s"
                  % (r["feature"], r["weight"], r["se"], r["z"], r["q"],
                     "yes" if r["sig"] else "-", r["level_zh"],
                     str(r["n_needed"]) if r["reachable"] else "不可达"))
        if tmpl is not None:
            print("\n解读:")
            for e in tmpl.explain(s["diagnostics"], df, cat_names=cat_names, trait_map=tmap):
                print("  · " + e["text"])

    if s.get("stability"):
        from hybridscope.stability import to_text
        print("\n" + to_text(est.stability_))

    if args.json:
        payload = dict(s)
        if prep is not None:
            payload["prep"] = prep.report()
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print("已写出 %s" % args.json)

    need_snap = bool(args.snapshot or args.compare)
    snap = None
    if need_snap or args.report:
        snap = est.snapshot(df=df_raw, name=args.name or profile, source=path,
                            schema=schema, profile=profile)
    if snap is not None and snap.get("fingerprint"):
        fp = snap["fingerprint"]
        print("\n数据指纹 hash_rows=%s  hash_pool=%s（%d 行 × %d 列）"
              % (fp["hash_rows"], fp["hash_pool"], fp["n_rows"], fp["n_cols"]))
    if args.snapshot and snap is not None:
        from hybridscope.monitor import save_snapshot
        save_snapshot(snap, args.snapshot)
        print("快照已写出 %s" % args.snapshot)
    if args.compare and snap is not None:
        from hybridscope.monitor import compare, load_snapshot
        old = load_snapshot(args.compare)
        rep = compare(old, snap, alpha=args.fdr_alpha)
        print("\n" + rep["verdict_text"])

    if args.report:
        from hybridscope.report import write_report
        explain = None
        if tmpl is not None and s.get("diagnostics"):
            explain = [e["text"] for e in
                       tmpl.explain(s["diagnostics"], df_raw, cat_names=cat_names,
                                   trait_map=tmap)]
        write_report(args.report, est, df=df_raw, source=path,
                     title="HybridScope 分析报告%s"
                           % (("（%s）" % (args.name or profile))
                              if (args.name or profile) else ""),
                     template=tmpl, template_explain=explain)
        print("报告已写出 %s" % args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
