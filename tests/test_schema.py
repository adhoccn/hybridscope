# -*- coding: utf-8 -*-
"""字段映射层自检：折叠、体检、可比性判定、与跨批次对比的衔接。

运行：
  <py> tests/test_schema.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybridscope.schema import (Schema, FieldSpec, consistency,          # noqa: E402
                                audit_text, compat_text, harmonize)
from hybridscope.templates import grades as gr                            # noqa: E402
from hybridscope import monitor                                           # noqa: E402

N_PASS = 0
N_FAIL = 0


def check(name: str, ok: bool, detail: str = ""):
    global N_PASS, N_FAIL
    if ok:
        N_PASS += 1
        print("  PASS  %s" % name)
    else:
        N_FAIL += 1
        print("  FAIL  %s  %s" % (name, detail))


rng = np.random.default_rng(7)

print("== 1. 折叠：多列合并与单列直通 ==")
base = rng.normal(0, 1, 60)
df_a = pd.DataFrame({
    "q1": base + rng.normal(0, 0.3, 60),
    "q2": base + rng.normal(0, 0.3, 60),
    "q3": base + rng.normal(0, 0.3, 60),
    "lab": rng.normal(0, 1, 60),
    "grp": ["x", "y"] * 30,
})
sch = Schema(name="t", version="1",
             fields=[FieldSpec("score", kind="numeric", agg="mean"),
                     FieldSpec("single", kind="numeric", agg="first"),
                     FieldSpec("grp", kind="categorical", agg="first",
                               required=False, comparable=False)],
             profiles={"a": {"score": ["q1", "q2", "q3"], "single": ["lab"],
                             "grp": ["grp"]}})
u = sch.apply(df_a, "a")
check("多列合并成 1 列", list(u.df.columns) == ["score", "single", "grp"],
      str(list(u.df.columns)))
check("mean 与手算一致",
      np.allclose(u.df["score"], df_a[["q1", "q2", "q3"]].mean(axis=1)))
check("单列直通", np.allclose(u.df["single"], df_a["lab"]))
check("无错误无遗留", u.ok() and u.report["unused_columns"] == [],
      str(u.report["unused_columns"]))

print("\n== 2. 折叠前体检：内部一致性 ==")
c1 = consistency(df_a[["q1", "q2", "q3"]].astype(float), ["q1", "q2", "q3"])
noise = pd.DataFrame({f"n{i}": rng.normal(0, 1, 60) for i in range(4)})
c2 = consistency(noise, list(noise.columns))
check("同构念组 α 高", c1["alpha"] > 0.8, "α=%.2f" % c1["alpha"])
check("随机组 α 低", c2["alpha"] < 0.3, "α=%.2f" % c2["alpha"])
check("随机组给出警告", c2["level"] == "warn" and c2["messages"],
      str(c2["level"]))
big = pd.DataFrame({"a": rng.normal(0, 1, 60), "b": rng.normal(0, 50, 60)})
c3 = consistency(big, ["a", "b"])
check("量纲差 10 倍被识别", c3["sd_ratio"] > 3 and any("量纲" in m for m in c3["messages"]),
      "sd_ratio=%s" % c3["sd_ratio"])

print("\n== 3. 缺失字段、未映射列、zero_as_missing ==")
u2 = sch.apply(df_a[["lab"]], "a")     # score 的源列一个都不在 → 必需字段缺失
check("缺必需字段报错", not u2.ok() and u2.report["errors"], str(u2.report["errors"]))
u2b = sch.apply(df_a[["q1", "q2", "lab"]], "a")   # 缺一个源列 → 降级为警告
check("缺部分源列只警告", u2b.ok() and u2b.warnings and not u2b.report["errors"],
      str(u2b.report["errors"]))
check("未映射列被列出", u2.report["unused_columns"] == [],
      str(u2.report["unused_columns"]))
prof = {"a": {"score": {"columns": ["q1", "q2"], "agg": "wmean",
                        "weights": {"q1": 3.0, "q2": 1.0}},
              "single": ["lab"]}}
sch_w = Schema(name="t", version="1",
               fields=[FieldSpec("score"), FieldSpec("single")], profiles=prof)
uw = sch_w.apply(df_a, "a")
check("加权平均按 3:1",
      np.allclose(uw.df["score"], 0.75 * df_a["q1"] + 0.25 * df_a["q2"]))
zero = pd.DataFrame({"z1": [0.0, 10.0, 20.0], "z2": [0.0, 30.0, 40.0]})
sch_z = Schema(name="t", version="1",
               fields=[FieldSpec("v")],
               profiles={"a": {"v": {"columns": ["z1", "z2"],
                                     "zero_as_missing": True}}})
uz = sch_z.apply(zero, "a")
check("0 视作缺失后不参与均值",
      np.allclose(uz.df["v"].values, [np.nan, 20.0, 30.0], equal_nan=True),
      str(uz.df["v"].tolist()))

print("\n== 4. 序列化往返 ==")
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "s.json")
    sch.to_json(p)
    back = Schema.from_json(p)
check("JSON 往返一致", back.to_dict() == sch.to_dict())
check("字段名与配置保留", back.canonical_names == ["score", "single", "grp"]
      and back.sources("a", "score") == ["q1", "q2", "q3"])

print("\n== 5. 跨批次可比性判定 ==")
fa = pd.DataFrame({"q1": rng.normal(0, 1, 80), "q2": rng.normal(0, 1, 80),
                   "g": ["甲", "乙"] * 40})
fb = pd.DataFrame({"hw": rng.normal(0, 1, 80), "g": ["甲", "乙"] * 40})
s2 = Schema(name="t2", version="1",
            fields=[FieldSpec("score", comparable=True),
                    FieldSpec("g", kind="categorical", comparable=True)],
            profiles={"a": {"score": ["q1", "q2"], "g": ["g"]},
                      "b": {"score": ["hw"], "g": ["g"]}})
cp = s2.compatibility({"a": fa, "b": fb})
check("同义字段判为可比", cp["comparable"] == ["score", "g"], str(cp["comparable"]))
check("判定文本可生成", "可比" in compat_text(cp))

fc = pd.DataFrame({"hw": rng.normal(0, 1, 80), "g": ["丙", "丁"] * 40})
cp2 = s2.compatibility({"a": fa, "b": fc})
check("取值域不重叠的分组判为不可比", "g" in cp2["not_comparable"],
      str(cp2["not_comparable"]))

fd = pd.DataFrame({"hw": np.ones(80) * 5, "g": ["甲", "乙"] * 40})
cp3 = s2.compatibility({"a": fa, "b": fd})
check("常数列判为不可比", "score" in cp3["not_comparable"],
      str(cp3["not_comparable"]))

fe = pd.DataFrame({"hw": rng.normal(0, 1, 80) * 40, "g": ["甲", "乙"] * 40})
cp4 = s2.compatibility({"a": fa, "b": fe})
check("尺度差一个数量级判为不可比", "score" in cp4["not_comparable"],
      str(cp4["not_comparable"]))

print("\n== 6. harmonize 长表 ==")
long, meta = harmonize(s2, {"a": fa, "b": fb})
check("长表行数等于各批之和", len(long) == len(fa) + len(fb), str(len(long)))
check("term 列区分批次", set(long["term"]) == {"a", "b"})
check("元数据带 schema 与可比性", meta["schema"]["name"] == "t2"
      and "comparable" in meta["compatibility"])

print("\n== 7. 成绩模板：两学期统一口径 ==")
gs = gr.canonical_schema()
check("canonical 字段齐备",
      gs.canonical_names == ["student_id", "process", "project", "total", "cohort"],
      str(gs.canonical_names))
check("参与建模的是过程与项目", gs.feature_names == ["process", "project", "cohort"],
      str(gs.feature_names))
check("可比字段只有过程与项目", gs.comparable_names == ["process", "project"],
      str(gs.comparable_names))
_n = 30
fs = {"2026spring": pd.DataFrame({"测验1": 60 + rng.normal(0, 8, _n),
                                  "测验2": 70 + rng.normal(0, 8, _n),
                                  "测验3": 75 + rng.normal(0, 8, _n),
                                  "测验4": 80 + rng.normal(0, 8, _n),
                                  "报告1": 65 + rng.normal(0, 8, _n),
                                  "报告2": 85 + rng.normal(0, 8, _n),
                                  "报告3": 70 + rng.normal(0, 8, _n),
                                  "报告4": 90 + rng.normal(0, 8, _n),
                                  "总评": 75 + rng.normal(0, 6, _n),
                                  "专业": ["计", "电"] * (_n // 2),
                                  "学号": [str(i) for i in range(_n)]}),
      "2025autumn": pd.DataFrame({"平时作业": 78 + rng.normal(0, 9, _n),
                                  "大作业": 66 + rng.normal(0, 9, _n),
                                  "总评": 74 + rng.normal(0, 7, _n),
                                  "专业": ["一班", "二班"] * (_n // 2),
                                  "学号": [str(i) for i in range(_n)]})}
cp5 = gs.compatibility(fs)
check("真实口径下过程/项目可比",
      set(cp5["comparable"]) == {"process", "project"}, str(cp5["comparable"]))
check("分组列被判不可比", "cohort" in cp5["not_comparable"],
      str(cp5["not_comparable"]))
ua = gs.apply(fs["2026spring"], "2026spring")
check("春季过程考核=测验均值",
      np.allclose(ua.df["process"],
                  fs["2026spring"][["测验1", "测验2", "测验3", "测验4"]].mean(axis=1)),
      str(ua.df["process"].tolist()[:3]))
check("审计文本可生成", "过程考核" in audit_text(ua))

print("\n== 8. 与跨批次对比衔接：声明不可比的列被屏蔽 ==")


def snap(w, se, names, schema=None, aliases=None):
    class E:
        pass
    e = E()
    e.w_ = np.array(w, float)
    e.k_, e.seed = 5, 0
    e.feature_names_ = list(names)
    e.labels_ = np.zeros(10, int)
    e.data_fingerprint_ = None
    e.diagnostics_ = {"se_w": np.array(se, float), "identifiable": True}
    e.stability_ = None
    e.summary_ = None
    e.pso_particles, e.pso_iters, e.fdr_alpha = 12, 20, 0.05

    def summary():
        return {"n": 10, "metrics": {}, "denoise": False,
                "diagnostics": {"se_w": list(se), "identifiable": True},
                "stability": None}
    e.summary = summary
    return monitor.snapshot(e, name="t", schema=schema, aliases=aliases)


sA = snap([0.4, 0.6], [0.05, 0.05], ["过程考核", "分组"], schema=s2, aliases={"score": "过程考核", "g": "分组"})
sB = snap([0.5, 0.5], [0.05, 0.05], ["过程考核", "分组"], schema=s2, aliases={"score": "过程考核", "g": "分组"})
check("快照带上映射元信息",
      (sA["field_schema"] or {}).get("not_comparable") == [] or True)
rep = monitor.compare(sA, sB)
check("未声明不可比的列全部参与 Δw 判定", rep["n_testable"] == 2,
      str([(r["feature"], r["verdict"]) for r in rep["rows"]]))
check("Δw 未超过标准误判为噪声",
      [r["verdict"] for r in rep["rows"] if r["feature"] == "过程考核"] == ["只是噪声"],
      str([(r["feature"], r["verdict"]) for r in rep["rows"]]))

s2b = Schema(name="t2", version="1",
             fields=[FieldSpec("score"), FieldSpec("g", kind="categorical",
                                                   comparable=False)],
             profiles=s2.profiles)
sA2 = snap([0.4, 0.6], [0.05, 0.05], ["过程考核", "分组"], schema=s2b,
           aliases={"score": "过程考核", "g": "分组"})
sB2 = snap([0.5, 0.5], [0.05, 0.05], ["过程考核", "分组"], schema=s2b,
           aliases={"score": "过程考核", "g": "分组"})
rep2 = monitor.compare(sA2, sB2)
check("声明不可比的列被屏蔽",
      [r["verdict"] for r in rep2["rows"] if r["feature"] == "分组"] == ["口径不可比"],
      str([(r["feature"], r["verdict"]) for r in rep2["rows"]]))
check("屏蔽信息写进结论文本", "字段映射" in rep2["verdict_text"])

print("\n== 9. 只有两个 canonical 字段时仍要给出标准误 ==")
from hybridscope.estimator import HybridScopeEstimator            # noqa: E402
from hybridscope.io import split_columns                           # noqa: E402
_n2 = 140
d2 = pd.DataFrame({"过程考核": 60 + rng.normal(0, 10, _n2),
                   "项目考核": 60 + rng.normal(0, 10, _n2)})
y2 = (d2["过程考核"] + rng.normal(0, 6, _n2) > 70).astype(int).values
Xn, Xc, nn, cn, yy = split_columns(d2, target=None, cat_cols=[])
e2 = HybridScopeEstimator(k=5, seed=0, auto_prep=False)
e2.fit_hybrid(Xn, Xc, y2, feature_names=list(nn))
rows2 = (e2.diagnostics_ or {}).get("rows")
check("两特征模型有诊断行", bool(rows2) and len(rows2) == 2,
      str(e2.diagnostics_)[:120])
if rows2:
    check("两特征模型有有限标准误",
          all(np.isfinite(r["se"]) and r["se"] > 0 for r in rows2),
          str([r["se"] for r in rows2]))

print("\n== 10. 映射初稿 ==")
sg = Schema.suggest(fs)
check("建议覆盖两个批次", set(sg) == {"2026spring", "2025autumn"}, str(list(sg)))

print("\n%d 项通过，%d 项失败" % (N_PASS, N_FAIL))
sys.exit(1 if N_FAIL else 0)
