# -*- coding: utf-8 -*-
"""生成可直接上传到工作台的建模 CSV（脱敏，且不含会泄漏标签的总评列）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from parse_grades import parse_2025autumn, parse_2026spring

DATA = Path(__file__).resolve().parent / "data"
SPRING_PDF = r"C:\baidunetdiskdownload\考题\2026春\人工智能导论\2-成绩单.pdf"
AUTUMN_XLSX = r"C:\baidunetdiskdownload\考题\2025秋\人工智能导论\2-总评成绩表(含过程).xlsx"


def band(s: pd.Series, cuts=(70, 85), names=("需关注", "合格", "优秀")) -> pd.Series:
    return pd.cut(s, bins=[-np.inf, *cuts, np.inf], labels=list(names)).astype(str)


def merge_tail(col: pd.Series, min_count: int = 3) -> pd.Series:
    vc = col.value_counts()
    return col.where(col.isin(vc[vc >= min_count].index), "其他")


def build(df: pd.DataFrame, num_cols: list[str], cat_name: str = "专业") -> pd.DataFrame:
    """去掉缺考行，压缩长尾类别，档位作标签，标签列放最后。"""
    work = df.copy()
    keep = work[num_cols].sum(axis=1) > 0          # 全零 = 未参加考核
    dropped = int((~keep).sum())
    work = work[keep]
    work[cat_name] = merge_tail(work["专业"].fillna("未标注"))
    work["学情档位"] = band(work["总评"])
    out = work[num_cols + [cat_name, "学情档位"]]
    print("  丢弃缺考行 %d，剩余 %d" % (dropped, len(out)))
    return out


def main() -> None:
    spring = build(parse_2026spring(SPRING_PDF),
                   ["测验1", "测验2", "测验3", "测验4",
                    "报告1", "报告2", "报告3", "报告4"])
    autumn = build(parse_2025autumn(AUTUMN_XLSX), ["平时作业", "大作业"],
                   cat_name="行政班")

    DATA.mkdir(parents=True, exist_ok=True)
    for name, df in [("upload_2026spring.csv", spring),
                     ("upload_2025autumn.csv", autumn)]:
        p = DATA / name
        df.to_csv(p, index=False, encoding="utf-8")
        print("%-24s %3d 行 x %d 列  %s" % (name, len(df), df.shape[1], p))
        print("   档位分布:", df["学情档位"].value_counts().to_dict())
        print("   %s 分布:" % df.columns[-2], df.iloc[:, -2].value_counts().to_dict())


if __name__ == "__main__":
    main()
