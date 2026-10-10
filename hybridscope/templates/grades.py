# -*- coding: utf-8 -*-
"""学情分型模板：把成绩表的字段语义固定下来，换一届学生可直接复用。

真实成绩表的坑：学号被当成数值特征、行政班被当成噪声、缺考的 0 分被当成
真实成绩、专业课成绩和公共课成绩量纲不同。模板一次性处理这些,
并把权重翻译成人能读的话（"过程性评价主导分型"而不是"x1=0.31"）。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from hybridscope.schema import FieldSpec, Schema

ID_HINTS = ("学号", "编号", "考号", "姓名", "考生号", "id", "ID")
CAT_HINTS = ("班", "专业", "学院", "性别", "学期", "年级", "层次", "科目", "课程")
TARGET_HINTS = ("总评", "总分", "总成绩", "期末成绩", "综合成绩", "档位")
# 缺考/缓考的占位值：看起来是数，其实是"没参加"
ABSENT_TOKENS = ("缺考", "缓考", "免修", "未交", "-", "—", "无", "")

TRAITS = [
    (("作业", "平时", "测验", "小测", "出勤", "课堂", "过程", "过程考核"), "过程性评价"),
    (("报告", "大作业", "项目", "设计", "实验", "论文", "项目考核"), "综合/项目产出"),
    (("期末", "期中", "统考", "结课"), "终结性考核"),
    (("上机", "机试", "实操", "编程"), "实操能力"),
    (("考勤", "签到"), "出勤"),
]
BAND_CUTS = (70, 85)
BAND_NAMES = ("需关注", "合格", "优秀")


def detect_roles(df: pd.DataFrame) -> dict:
    """按列名猜角色：id / cat / num / target。"""
    roles = {}
    for c in df.columns:
        name = str(c)
        low = name.lower()
        if any(h in name for h in ID_HINTS):
            roles[c] = "id"
        elif any(h in name for h in CAT_HINTS) or not pd.api.types.is_numeric_dtype(df[c]):
            roles[c] = "cat"
        elif any(h in name for h in TARGET_HINTS):
            roles[c] = "target"
        elif any(k in low for k in ("score", "grade")):
            roles[c] = "target"
        else:
            roles[c] = "num"
    return roles


def pick_target(df: pd.DataFrame, roles: dict | None = None) -> str | None:
    roles = roles or detect_roles(df)
    for c, r in roles.items():
        if r == "target":
            return c
    return None


def feature_split(df: pd.DataFrame, target: str | None = None,
                  roles: dict | None = None) -> tuple[list[str], list[str]]:
    roles = roles or detect_roles(df)
    num = [c for c, r in roles.items() if r == "num" and c != target]
    cat = [c for c, r in roles.items() if r == "cat" and c != target]
    return num, cat


def clean_absent(s: pd.Series) -> pd.Series:
    """缺考/缓考等占位值转 NaN，别让 0 分混进成绩分布。"""
    if pd.api.types.is_numeric_dtype(s):
        return s
    v = s.astype(str).str.strip()
    return pd.to_numeric(v.where(~v.isin(ABSENT_TOKENS)), errors="coerce")


def band(s: pd.Series, cuts=BAND_CUTS, names=BAND_NAMES) -> pd.Series:
    """成绩 → 学情档位（需关注 / 合格 / 优秀）。"""
    return pd.cut(s, bins=[-np.inf, *cuts, np.inf], labels=list(names)).astype(str)


def trait_of(col: str) -> str:
    """按最长匹配归类（"大作业"要归到项目产出，而不是被"作业"抢走）。"""
    best, blen = None, 0
    for keys, trait in TRAITS:
        for k in keys:
            if k in str(col) and len(k) > blen:
                best, blen = trait, len(k)
    return best or "未归类考核项"


def explain(rows: list[dict], df: pd.DataFrame | None = None,
            cat_names: list[str] | None = None,
            trait_map: dict | None = None) -> list[dict]:
    """把权重行翻译成一句话。rows 为 diagnostics 的 rows。

    trait_map：字段名 → 中文含义。用 canonical 字段建模时列名是英文
    （process/project），trait_of 认不出来，映射里的 label 正好补上。
    """
    cat_set = set(cat_names or [])
    tmap = trait_map or {}
    out = []
    for r in rows:
        name = str(r.get("feature", ""))
        w = float(r.get("weight", 0.0))
        z = float(r.get("z", 0.0))
        lvl = r.get("level_fdr", r.get("level", "red"))   # 优先用 FDR 校正后档位
        q = r.get("q", None)
        sig = bool(r.get("sig", False))
        trait = ("分组变量" if name in cat_set
                 else tmap.get(name) or trait_of(name))
        if name in cat_set:
            tail = {"green": "该分组变量确实区分出不同的学情群体",
                    "amber": "区分方向可信但精度不足",
                    "red": "看不出该分组变量对学情分型有作用"}.get(
                        lvl, "权重落在边界上，等价于该分组变量未参与分型")
        else:
            if lvl == "green":
                tail = "该考核项对分型的贡献可被数据支持"
            elif lvl == "amber":
                tail = "方向可信但精度不足，需更多样本"
            elif lvl == "red":
                tail = ("多重检验校正后不显著（q=%s），"
                        "当前样本量下无法把它和“各考核项等权”区分开"
                        % ("%.3f" % q if isinstance(q, float) else "n/a"))
            else:
                tail = "权重落在单纯形边界上，等价于该项未参与分型"
        share = ""
        if rows:
            tot = sum(abs(float(x.get("weight", 0.0))) for x in rows) or 1.0
            share = "占权重 %.0f%%，" % (abs(w) / tot * 100)
        qs = ("，q=%.3f%s" % (q, "（显著）" if sig else "")) if isinstance(q, float) else ""
        out.append({
            "feature": name,
            "trait": trait,
            "text": "%s（%s）权重 %.3f，%sz=%.2f%s：%s。"
                    % (name, trait, w, share, z, qs, tail),
        })
    return out


def summarize(df: pd.DataFrame, target: str | None = None,
              roles: dict | None = None) -> dict:
    """成绩表的语义概览：角色划分、缺考情况、各考核项分布。"""
    roles = roles or detect_roles(df)
    target = target or pick_target(df, roles)
    num, cat = feature_split(df, target=target, roles=roles)
    absent = {}
    for c in num:
        v = clean_absent(df[c])
        absent[c] = int(v.isna().sum())
    return {
        "n_students": int(len(df)),
        "target": target,
        "numeric_items": num,
        "categorical_items": cat,
        "id_columns": [c for c, r in roles.items() if r == "id"],
        "absent_counts": absent,
        "traits": {c: trait_of(c) for c in num},
    }


def drop_absent_rows(df: pd.DataFrame, target: str | None = None,
                     roles: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """剔除缺考/未参与行：所有考核项都是 0（或缺失）且总评也是 0。

    这类行不是"成绩很差的学生"，是"没参加"，留在里面会把分布拉向 0。
    """
    roles = roles or detect_roles(df)
    num, _ = feature_split(df, target=target, roles=roles)
    M = df[num].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    flag = (M.abs().sum(axis=1) == 0)
    keep = ~flag
    # 只有当标签列本身是分数时才用「总评=0」判缺考；标签已经是文字档位
    # （学情档位=优秀/合格）时不能这么判，否则会把所有人都当成缺考剔掉。
    if target is not None and target in df.columns:
        t = pd.to_numeric(df[target], errors="coerce")
        if t.notna().any():
            flag = flag | (t.fillna(0.0) == 0)
            keep = ~flag
    return (df[keep].reset_index(drop=True),
            {"dropped_absent": int((~keep).sum()), "kept": int(keep.sum())})


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path) if str(path).lower().endswith(".csv") else pd.read_excel(path)
    df.columns = [re.sub(r"^﻿", "", str(c)).strip() for c in df.columns]
    return df


# ---------------------------------------------------------------- 统一口径

# 两学期成绩表字段对不齐，是因为教务系统的分项粒度变了：
#   2026 春：测验 1–4（过程，占 60%）+ 报告 1–4（项目，占 40%）
#   2025 秋：平时作业（过程，占 60%）+ 大作业（项目，占 40%）
# 两边对总评的加权结构一致（用总评对分项做最小二乘可验证：春 0.150/0.099×4，
# 秋 0.599/0.400），所以过程考核与项目考核是一对可对齐的 canonical 字段。
# 注意「专业」不能对齐：春季填的是专业名（选修课，学生来自各专业），
# 秋季那一列填的是行政班（计算机 25-x 班），同名不同义，标 comparable=False。

CANONICAL_FIELDS = [
    FieldSpec(name="student_id", label="学号", kind="categorical", agg="first",
              role="id", required=False, comparable=False,
              note="脱敏后的学号哈希，只用于核对，不进模型"),
    FieldSpec(name="process", label="过程考核得分", kind="numeric", agg="mean",
              role="feature", comparable=True,
              note="平时过程性考核的百分制得分：春=测验 1–4 均值，秋=平时作业"),
    FieldSpec(name="project", label="项目考核得分", kind="numeric", agg="mean",
              role="feature", comparable=True,
              note="综合/项目类考核的百分制得分：春=报告 1–4 均值，秋=大作业"),
    FieldSpec(name="total", label="总评", kind="numeric", agg="first",
              role="target", required=False, comparable=False,
              note="加权总评，只用于切分学情档位；作特征会泄漏标签"),
    FieldSpec(name="cohort", label="群体分组", kind="categorical", agg="first",
              role="feature", required=False, comparable=False,
              note="春=专业名，秋=行政班，语义不同，跨学期不作权重比较"),
]

DEFAULT_PROFILES = {
    "2026spring": {
        "student_id": ["学号"],
        "process": ["测验1", "测验2", "测验3", "测验4"],
        "project": ["报告1", "报告2", "报告3", "报告4"],
        "total": ["总评"],
        "cohort": ["专业"],
    },
    "2025autumn": {
        "student_id": ["学号"],
        "process": ["平时作业"],
        "project": ["大作业"],
        "total": ["总评"],
        "cohort": ["专业"],
    },
}


def canonical_schema(profiles: dict | None = None, version: str = "1") -> Schema:
    """学情分型的统一口径：过程考核 / 项目考核 / 总评 / 分组。"""
    return Schema(
        name="grades-canonical",
        version=version,
        note="学情分型统一口径：过程性考核与项目类考核各一个百分制得分，"
             "两学期对总评的权重结构一致（60/40），可跨学期比较；"
             "分组列（专业/行政班）语义不同，不参与跨学期权重比较。",
        fields=list(CANONICAL_FIELDS),
        profiles=profiles or {k: dict(v) for k, v in DEFAULT_PROFILES.items()},
    )


def unify(frames: dict[str, pd.DataFrame], schema: Schema | None = None,
          term_col: str = "term") -> tuple[pd.DataFrame, dict]:
    """把多个学期的成绩表对齐到统一口径，返回长表 + 审计/可比性结论。"""
    from hybridscope.schema import harmonize
    schema = schema or canonical_schema()
    return harmonize(schema, frames, term_col=term_col)
