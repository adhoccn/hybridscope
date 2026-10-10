# -*- coding: utf-8 -*-
"""v2.3 自检：问卷量表与设备运维两个新模板。

运行：
  <py> tests/test_templates.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybridscope.templates import get, names                       # noqa: E402
from hybridscope.templates import survey, ops, grades              # noqa: E402

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


rng = np.random.default_rng(3)

print("== 1. 模板注册表 ==")
check("三个模板都注册了", set(names()) == {"grades", "survey", "ops"}, str(names()))
check("按名可取", get("survey") is survey and get("ops") is ops and get("grades") is grades)

print("\n== 2. 问卷模板：反向计分 / 直线作答 / 维度合成 ==")
s = pd.Series([1, 3, 5], dtype=float)
check("反向计分 1↔5", list(survey.reverse_score(s, (1, 5))) == [5.0, 3.0, 1.0],
      str(list(survey.reverse_score(s, (1, 5)))))

n = 120
eng = rng.normal(3.5, 0.6, n)
anx = rng.normal(2.8, 0.7, n)
df = pd.DataFrame({"问卷编号": [f"R{i:03d}" for i in range(n)]})
for i in range(1, 5):
    df[f"Q{i}"] = np.clip(np.round(eng + rng.normal(0, 0.3, n)), 1, 5)
for i in range(5, 9):
    df[f"Q{i}"] = np.clip(np.round(anx + rng.normal(0, 0.3, n)), 1, 5)
df["Q4"] = 6 - df["Q4"]                       # 反向题
df["性别"] = ["男", "女"] * (n // 2)
df["档位"] = ["高" if e > 3.5 else "低" for e in eng]
df.loc[0:2, [f"Q{i}" for i in range(1, 9)]] = 3     # 3 份直线作答

dims = {"学习投入": [f"Q{i}" for i in range(1, 5)],
        "学习焦虑": [f"Q{i}" for i in range(5, 9)]}
roles = survey.detect_roles(df)
check("问卷编号判为标识", roles.get("问卷编号") == "id", str(roles.get("问卷编号")))
check("题目判为数值特征", all(roles.get(f"Q{i}") == "num" for i in range(1, 9)))
check("档位判为标签", survey.pick_target(df, roles) in (None, "档位"),
      str(survey.pick_target(df, roles)))

# 反向题不反转 → 维度内互相抵消，α 会明显偏低
sc_bad, _ = survey.dimension_scores(df, dims, reverse=[], scale=(1, 5))
sc_ok, audit_ok = survey.dimension_scores(df, dims, reverse=["Q4"], scale=(1, 5))
a_bad = survey.dimension_scores(df, dims, reverse=[], scale=(1, 5))[1]
a_bad = a_bad["dimensions"]["学习投入"]["alpha"]
a_ok = audit_ok["dimensions"]["学习投入"]["alpha"]
check("反转反向题后 α 提升", a_ok > a_bad + 0.2, "反转 %.2f vs 未反转 %.2f" % (a_ok, a_bad))
check("同构念维度 α 高", a_ok > 0.8, "α=%.2f" % a_ok)
check("维度分均值落在量表范围内",
      1 <= sc_ok["学习投入"].mean() <= 5 and 1 <= sc_ok["学习焦虑"].mean() <= 5,
      "%.2f / %.2f" % (sc_ok["学习投入"].mean(), sc_ok["学习焦虑"].mean()))

flag = survey.straight_lining(df, [f"Q{i}" for i in range(1, 9)])
check("检出直线作答", int(flag.sum()) >= 3, str(int(flag.sum())))
left, dropped = survey.drop_invalid(df, [f"Q{i}" for i in range(1, 9)])
check("剔除直线作答", dropped == int(flag.sum()) and len(left) == n - dropped,
      "%d / %d" % (dropped, len(left)))

summ = survey.summarize(df, target="档位", roles=roles)
check("问卷概览给出份数与题数", summ["n_respondents"] == n and summ["n_items"] == 8,
      str(summ))
check("概览报出直线作答数", summ["straight_line"] == int(flag.sum()),
      str(summ["straight_line"]))
cav = survey.caveats(summ, audit_ok)
check("注意事项提到直线作答", any("直线作答" in c for c in cav), str(cav))

rows = [{"feature": "学习投入", "weight": 0.7, "z": 3.1, "q": 0.001, "sig": True,
         "level_fdr": "green"},
        {"feature": "性别", "weight": 0.0, "z": 0.0, "q": 1.0, "sig": False,
         "level_fdr": "boundary"}]
ex = survey.explain(rows, cat_names=["性别"])
check("维度被认成投入度", ex[0]["trait"] == "投入度", str(ex[0]))
check("人群列被认成分组变量", ex[1]["trait"] == "分组变量", str(ex[1]))

print("\n== 3. 运维模板：重复测量 / 不平衡 / 量纲 ==")
m = 240
op = pd.DataFrame({
    "设备编号": [f"D{i:02d}" for i in range(60)] * 4,      # 每台采 4 次
    "温度": rng.normal(60, 8, m),
    "振动加速度": rng.normal(0, 3000, m),
    "电流": rng.normal(12, 2, m),
    "型号": ["A型", "B型"] * (m // 2),
    "状态": np.where(rng.random(m) < 0.07, "故障", "正常"),
})
r2 = ops.detect_roles(op)
check("状态判为标签而不是分组", r2.get("状态") == "target", str(r2.get("状态")))
check("设备编号判为标识", r2.get("设备编号") == "id", str(r2.get("设备编号")))
check("型号判为分组", r2.get("型号") == "cat", str(r2.get("型号")))
check("通道判为数值特征", r2.get("温度") == "num" and r2.get("电流") == "num")

bal = ops.class_balance(op, "状态")
check("识别出类别不平衡", bal["imbalanced"] and bal["minority_share"] < 0.10,
      str(bal["minority_share"]))
rep = ops.repeated_units(op)
check("识别出重复测量", (not rep["independent"]) and rep["max_repeat"] == 4,
      str(rep))
check("重复单元数正确", rep["n_units"] == 60 and rep["n_rows"] == m, str(rep))

sm = ops.summarize(op, roles=r2)
check("概览含量纲差", sm["scale_spread"] > 100, str(sm["scale_spread"]))
cv = ops.caveats(sm)
check("提示重复测量", any("不独立" in c for c in cv), str(cv))
check("提示类别不平衡", any("不平衡" in c for c in cv), str(cv))
check("提示量纲", any("量纲" in c for c in cv), str(cv))
check("通道语义归类", ops.trait_of("振动加速度") == "机械振动", ops.trait_of("振动加速度"))
ex2 = ops.explain([{"feature": "温度", "weight": 0.5, "z": 4.0, "q": 0.0,
                    "sig": True, "level_fdr": "green"}])
check("运维解读给出热状态", "热状态" in ex2[0]["text"], ex2[0]["text"])

print("\n== 4. 体检：量表不该按重复行剔除 ==")
from hybridscope.prep import Prep                                   # noqa: E402
wide = pd.DataFrame({"d1": np.round(rng.uniform(1, 5, 200)),
                     "d2": np.round(rng.uniform(1, 5, 200)),
                     "y": rng.integers(0, 2, 200)})
p1 = Prep()
out1 = p1.fit_transform(wide, target="y")
check("离散表保留重复行（多人同答）", len(out1) == len(wide),
      "%d → %d" % (len(wide), len(out1)))
cont = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200),
                     "y": rng.integers(0, 2, 200)})
cont = pd.concat([cont, cont.iloc[:20]])           # 人为的 20 行重复录入
p2 = Prep()
out2 = p2.fit_transform(cont, target="y")
check("连续表仍剔除重复行", len(out2) == len(cont) - 20,
      "%d → %d" % (len(cont), len(out2)))
p3 = Prep(drop_dup_rows=False)
out3 = p3.fit_transform(cont, target="y")
check("显式关闭时不剔重", len(out3) == len(cont), str(len(out3)))

print("\n%d 项通过，%d 项失败" % (N_PASS, N_FAIL))
sys.exit(1 if N_FAIL else 0)
