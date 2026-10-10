# -*- coding: utf-8 -*-
"""P0 自检：数据体检与 τ̂ 判据。

跑法：python tests/test_prep_advisor.py
不需要外部数据，全部用构造数据，检查四件事：
  1. 脏表（缺失/离群/低频档/标识列/量纲失衡）被正确修补
  2. 真平滑时序（AR(1) ρ=0.98）→ τ̂ 高 → 建议开去噪
  3. 按类分块存放的表 → 排序键检验判定阻塞 → 建议关去噪
  4. 纯白噪表 → 样本轴无结构 → 建议关去噪
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hybridscope.advisor import block_score, profile_ratio, recommend
from hybridscope.prep import Prep

FAIL = []


def check(name: str, cond: bool, detail: str = ""):
    print("%s %s%s" % ("PASS" if cond else "FAIL", name,
                       "" if cond else "  → " + detail))
    if not cond:
        FAIL.append(name)


def dirty_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "学号": ["S%05d" % i for i in range(n)],
        "成绩A": rng.normal(70, 10, n),
        "成绩B": rng.normal(70, 10, n) * 1000,      # 量级差 1000 倍
        "班级": rng.choice(["一班", "二班", "三班"], n),
        "生源": ["甲省"] * (n - 9) + ["乙省"] * 8 + ["丙省"] * 1,
        "常数项": np.ones(n),
        "几乎全缺": [np.nan] * int(0.9 * n) + list(rng.normal(0, 1, n - int(0.9 * n))),
    })
    k = max(1, n // 6)
    df.loc[rng.choice(n, k, replace=False), "成绩A"] = np.nan
    df.loc[rng.choice(n, max(1, k // 10), replace=False), "成绩B"] = 99999.0   # 离群
    df.loc[rng.choice(n, max(1, k // 3), replace=False), "班级"] = None
    return pd.concat([df, df.iloc[:3]], ignore_index=True)       # 3 条重复行


def test_prep():
    df = dirty_frame()
    p = Prep()
    out = p.fit_transform(df)
    rep = p.report()
    check("重复行被剔除", rep["n_dup_rows_dropped"] == 3, str(rep["n_dup_rows_dropped"]))
    dropped = {d["column"]: d["reason"] for d in rep["dropped"]}
    check("标识列被剔除", "学号" in dropped, str(list(dropped)))
    check("常量列被剔除", "常数项" in dropped, str(list(dropped)))
    check("高缺失列被剔除", "几乎全缺" in dropped, str(list(dropped)))
    check("缺失被填补而非整行丢弃", len(out) == len(df) - 3,
          "%d vs %d" % (len(out), len(df) - 3))
    check("修补后无缺失", int(out.isna().sum().sum()) == 0)
    check("量纲失衡触发标准化", rep["scaled"] is True)
    sd = out["成绩B"].std()
    check("标准化后量级对齐", 0.5 < sd < 2.0, "sd=%.3f" % sd)
    check("低频档并入其他（丙省 2 例）",
          set(out["生源"]) == {"甲省", "乙省", "其他"}, str(set(out["生源"])))
    check("类别缺失独立成档", "缺失" in set(out["班级"]), str(set(out["班级"])))
    # transform 可复用到新数据
    new = dirty_frame(n=60, seed=7)
    out2 = p.transform(new)
    check("transform 对新表列一致", list(out2.columns) == list(out.columns),
          "%s vs %s" % (list(out2.columns), list(out.columns)))


def smooth_series(n=400, rho=0.98, seed=1):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = rho * s[t - 1] + e[t]
    return s


def test_tau():
    n = 400
    # 1) 真平滑时序
    X = np.column_stack([smooth_series(n, 0.98, s) for s in range(3)])
    r = recommend(X)
    tau = np.nanmedian(profile_ratio(X))
    check("平滑时序 τ̂ ≥ 0.8", tau >= 0.8, "τ̂=%.3f" % tau)
    check("平滑时序建议开去噪", r["denoise"] is True, r["reason"])

    # 2) 按类分块存放：块内常数 + 块间跳变
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(4), n // 4)
    block = rng.normal(0, 1, 4)[groups]
    Xb = np.column_stack([block + rng.normal(0, 0.05, n) for _ in range(3)])
    bs = block_score(groups)
    check("分块存放被排序键检验识别", bs["blocked"] is True, str(bs))
    rb = recommend(Xb, groups=groups)
    check("分块存放建议关去噪", rb["denoise"] is False, rb["reason"])

    # 3) 白噪
    Xw = rng.normal(0, 1, (n, 3))
    rw = recommend(Xw)
    check("白噪表建议关去噪", rw["denoise"] is False, rw["reason"])
    check("白噪判据为无结构", rw["rule"] == "no-axis-structure", rw["rule"])


def main():
    print("== 数据体检 ==")
    test_prep()
    print("\n== τ̂ 判据 ==")
    test_tau()
    print("\n%s（%d 项未通过）" % ("全部通过" if not FAIL else "存在失败", len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
