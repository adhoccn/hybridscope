# -*- coding: utf-8 -*-
"""把两个学期的成绩表对齐到同一套 canonical 字段，产出统一口径长表。

用法：python cases/learning_analytics/make_unified.py

产出：
  data/grades_unified.csv   长表，含 term 列，两学期可直接一起建模或对比
  schema_grades.json        映射规则存档（可人工修订后重跑）
并在屏幕打印：字段映射审计 + 跨学期可比性结论。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hybridscope.schema import Schema, audit_text, compat_text
from hybridscope.templates import grades as gr

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PROFILES = {"2026spring": "grades_2026spring.csv",
            "2025autumn": "grades_2025autumn.csv"}


def load_raw(profile: str) -> pd.DataFrame:
    df = gr.load(DATA / PROFILES[profile])
    roles = gr.detect_roles(df)
    target = gr.pick_target(df, roles)
    df, rep = gr.drop_absent_rows(df, target=target, roles=roles)
    print("%s：剔除缺考/未参与 %d 人，余 %d 人"
          % (profile, rep["dropped_absent"], rep["kept"]))
    return df


def main() -> None:
    frames = {p: load_raw(p) for p in PROFILES}
    schema = gr.canonical_schema()
    schema.to_json(str(HERE / "schema_grades.json"))

    for p, df in frames.items():
        print()
        print(audit_text(schema.apply(df, p)))

    cp = schema.compatibility(frames)
    print()
    print(compat_text(cp))

    # 长表：统一口径 + 档位标签 + 便于核对的辅助列
    long, meta = gr.unify(frames, schema=schema)
    long["学情档位"] = gr.band(long["total"])
    # 分组语义不同，另存一列说明这一行的分组到底是专业还是行政班
    long["分组口径"] = np.where(long["term"] == "2026spring", "专业", "行政班")
    cols = ["term", "student_id", "process", "project", "total",
            "cohort", "分组口径", "学情档位"]
    long = long[cols].rename(columns={"term": "学期", "student_id": "学号",
                                      "process": "过程考核", "project": "项目考核",
                                      "total": "总评", "cohort": "分组"})
    out = DATA / "grades_unified.csv"
    long.to_csv(out, index=False, encoding="utf-8-sig")

    print()
    print("已写出 %s（%d 行）" % (out.name, len(long)))
    print("已写出 schema_grades.json")
    print()
    print(long.groupby("学期")[["过程考核", "项目考核", "总评"]]
          .agg(["count", "mean", "std"]).round(2).to_string())
    print()
    print("档位分布：")
    print(pd.crosstab(long["学期"], long["学情档位"]).to_string())

    # 口径核对：总评能否由两个 canonical 得分线性还原（权重结构是否一致）
    for term, g in long.groupby("学期"):
        X = np.c_[g["过程考核"], g["项目考核"], np.ones(len(g))]
        b, *_ = np.linalg.lstsq(X, g["总评"].values, rcond=None)
        resid = g["总评"].values - X @ b
        print("\n%s 总评 ≈ %.3f×过程考核 + %.3f×项目考核 %+.3f（残差 sd=%.3f）"
              % (term, b[0], b[1], b[2], resid.std()))


if __name__ == "__main__":
    main()
