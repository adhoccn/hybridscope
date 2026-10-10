# -*- coding: utf-8 -*-
"""从教务成绩表解析出脱敏建模数据：姓名剔除，学号单向哈希。"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

NUM = re.compile(r"^\d+(?:\.\d+)?$")
OUT = Path(__file__).resolve().parent / "data"


CLASS_RE = re.compile(r"([\u4e00-\u9fa5]+)\s*\d{2}-\d+\s*[\u4e00-\u9fa5]*班")


def _anon(sid: str) -> str:
    return hashlib.sha256(("hybridscope:" + str(sid)).encode()).hexdigest()[:10]


def _class_of(tail: str) -> str:
    """从姓名残片与班级混排的尾部里取出「专业 年级-序号 班」。"""
    m = CLASS_RE.findall(tail)
    return m[-1] if m else ""


def _major(cls_name: str, is_fu: bool) -> str:
    if not cls_name:
        return "辅修" if is_fu else "未标注"
    return re.split(r"\s+", cls_name.strip())[0]


def _parse_row(line: str):
    """一行 = 序号 学号 姓名 [行政班] + 11 个分数。"""
    toks = line.split()
    nums, i = [], len(toks)
    while i > 0 and len(nums) < 11:
        i -= 1
        if NUM.match(toks[i]):
            nums.append(float(toks[i]))
        else:
            break
    if len(nums) != 11:
        return None
    head = toks[:i]
    if len(head) < 3 or not head[0].isdigit():
        return None
    nums = nums[::-1]
    proc, lab = sum(nums[0:4]) * 0.15, sum(nums[5:9]) * 0.10
    if abs(proc - nums[4]) > 0.6 or abs(lab - nums[9]) > 0.6:
        return None
    return {"学号": _anon(head[1]), "行政班": _class_of(" ".join(head[3:])),
            "辅修": str(head[1]).upper().startswith("FX"),
            "测验1": nums[0], "测验2": nums[1], "测验3": nums[2], "测验4": nums[3],
            "报告1": nums[5], "报告2": nums[6], "报告3": nums[7], "报告4": nums[8],
            "总评": nums[10]}


def parse_2026spring(pdf: str) -> pd.DataFrame:
    reader = PdfReader(pdf)
    rows, buf = [], ""
    for pg in reader.pages:
        for raw in (pg.extract_text() or "").split("\n"):
            line = raw.strip()
            if not line:
                continue
            cand = (buf + " " + line).strip() if buf else line
            got = _parse_row(cand)
            if got:
                rows.append(got)
                buf = ""
            else:
                buf = cand if re.match(r"^\d+\s+\S{6,}", line) else ""
    df = pd.DataFrame(rows)
    df["专业"] = [_major(c, f) for c, f in zip(df["行政班"], df["辅修"])]
    return df.drop(columns=["辅修"])


def parse_2025autumn(xlsx: str) -> pd.DataFrame:
    raw = pd.read_excel(xlsx, header=None)
    body = raw.iloc[6:153]
    recs = []
    for _, r in body.iterrows():
        if pd.isna(r.iloc[1]):
            continue
        recs.append({"学号": _anon(r.iloc[1]), "行政班": str(r.iloc[3]),
                     "辅修": False,
                     "平时作业": float(r.iloc[4]), "大作业": float(r.iloc[6]),
                     "总评": float(r.iloc[8])})
    df = pd.DataFrame(recs)
    df["专业"] = [_major(c, f) for c, f in zip(df["行政班"], df["辅修"])]
    return df.drop(columns=["辅修"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spring = parse_2026spring(r"C:\baidunetdiskdownload\考题\2026春\人工智能导论\2-成绩单.pdf")
    autumn = parse_2025autumn(r"C:\baidunetdiskdownload\考题\2025秋\人工智能导论\2-总评成绩表(含过程).xlsx")

    for name, df in [("grades_2026spring.csv", spring), ("grades_2025autumn.csv", autumn)]:
        df.to_csv(OUT / name, index=False, encoding="utf-8-sig")
        print("%s  %d 行  %d 列" % (name, len(df), df.shape[1]))

    print("\n2026春 专业分布:\n", spring["专业"].value_counts().to_string())
    print("\n2026春 总评: mean=%.1f min=%.0f max=%.0f"
          % (spring["总评"].mean(), spring["总评"].min(), spring["总评"].max()))
    print("全零(缺考)行:", int(((spring[["测验1", "测验2", "测验3", "测验4"]].sum(axis=1) == 0)
                             & (spring[["报告1", "报告2", "报告3", "报告4"]].sum(axis=1) == 0)).sum()))
    zero = (spring[["测验1", "测验2", "测验3", "测验4", "报告1", "报告2", "报告3", "报告4"]] == 0)
    print("各分项 0 分占比:\n", (zero.mean() * 100).round(1).to_string())
    print("\n2025秋 总评: mean=%.1f min=%.0f max=%.0f"
          % (autumn["总评"].mean(), autumn["总评"].min(), autumn["总评"].max()))
    print("\n已脱敏：姓名不落盘，学号为 SHA256 前 10 位。")


if __name__ == "__main__":
    main()
