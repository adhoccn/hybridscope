# -*- coding: utf-8 -*-
"""问卷/量表模板：李克特题目 → 维度分 → 分型。

量表数据看着最简单（每题一个 1–5 的整数），但最容易做出假结论：
  1) 反向计分题没反转，维度分会自我抵消，α 会难看到像数据有问题；
  2) 直线作答（所有题都选同一个选项）是无效问卷，留着会凭空造出一个"簇"；
  3) 维度不是单题：把 20 道题直接当 20 个特征，权重解释不清，
     标准误也会因为维度内相关而失真。本模板先按维度合成，再进模型；
  4) 合成之前必须查内部一致性：α 低说明这几题测的不是同一个东西，
     平均出来的分数含义模糊（复用 schema.consistency）。

用法：
    dims = {"学习投入": ["Q1", "Q2", "Q3"], "学习焦虑": ["Q4", "Q5", "Q6"]}
    scores, audit = dimension_scores(df, dims, reverse=["Q4"], scale=(1, 5))
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from hybridscope.schema import consistency

ID_HINTS = ("问卷", "编号", "受访", "学号", "工号", "id", "ID", "respondent")
CAT_HINTS = ("性别", "年级", "专业", "学院", "学历", "岗位", "地区", "组别",
             "类别", "群体")
TARGET_HINTS = ("总分", "总均分", "类别", "档位", "分组", "标签", "score", "label")
SCALE_MIN, SCALE_MAX = 1, 5

# 维度名的语义归类
TRAITS = [
    (("投入", "参与", "engagement"), "投入度"),
    (("焦虑", "压力", "anxiety", "stress"), "情绪负荷"),
    (("动机", "兴趣", "motivation"), "动机"),
    (("策略", "方法", "strategy"), "学习策略"),
    (("自我", "效能", "self"), "自我效能"),
    (("满意", "评价", "satisf"), "满意度"),
    (("支持", "帮助", "support"), "外部支持"),
]


def detect_roles(df: pd.DataFrame) -> dict:
    """题目列（Q1..Qn / 含"题"/"item"）算数值特征，人口学信息算分组。"""
    roles = {}
    for c in df.columns:
        name = str(c)
        low = name.lower()
        if any(h in name for h in ID_HINTS):
            roles[c] = "id"
        elif any(h in name for h in TARGET_HINTS):
            roles[c] = "target"
        elif any(h in name for h in CAT_HINTS) or not pd.api.types.is_numeric_dtype(df[c]):
            roles[c] = "cat"
        elif re.match(r"^(q|Q|item|题)\s*\d+", name) or "题" in name or "item" in low:
            roles[c] = "num"
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


def reverse_score(s: pd.Series, scale: tuple[int, int] = (SCALE_MIN, SCALE_MAX)
                  ) -> pd.Series:
    """反向计分：lo + hi − x。不反转的话维度内题目会互相抵消。"""
    lo, hi = float(scale[0]), float(scale[1])
    return lo + hi - pd.to_numeric(s, errors="coerce")


def straight_lining(df: pd.DataFrame, cols: list[str],
                    tol: int = 0) -> pd.Series:
    """直线作答标记：所有题的作答极差 <= tol（几乎全选同一个选项）。"""
    M = df[cols].apply(pd.to_numeric, errors="coerce")
    rng = M.max(axis=1) - M.min(axis=1)
    return rng <= tol


def dimension_scores(df: pd.DataFrame, dims: dict[str, list[str]],
                     reverse: list[str] | None = None,
                     scale: tuple[int, int] = (SCALE_MIN, SCALE_MAX),
                     alpha_min: float = 0.60) -> tuple[pd.DataFrame, dict]:
    """按维度合成得分，并给出每个维度的一致性审计。

    反向题先反转，再取均值；α 低于 alpha_min 的维度会被标记——不是说不能
    用，而是要在结论里说明这个维度分数掺着噪声。
    """
    rev = set(reverse or [])
    out = pd.DataFrame(index=df.index)
    audit = {"dimensions": {}, "warnings": [], "scale": list(scale)}
    for dim, cols in dims.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            audit["warnings"].append("维度 %s 的题一个都不在表里，已跳过" % dim)
            continue
        M = pd.DataFrame(index=df.index)
        for c in cols:
            M[c] = (reverse_score(df[c], scale) if c in rev
                    else pd.to_numeric(df[c], errors="coerce"))
        # 量纲校验：整份量表应当共用同一个计分范围
        lo, hi = float(scale[0]), float(scale[1])
        bad = [c for c in cols
               if pd.to_numeric(M[c], errors="coerce").dropna().min() < lo - 1e-9
               or pd.to_numeric(M[c], errors="coerce").dropna().max() > hi + 1e-9]
        if bad:
            audit["warnings"].append(
                "维度 %s 的 %s 有超出 %g–%g 的取值：计分尺度可能不一致，先核对问卷版本"
                % (dim, "、".join(bad), lo, hi))
        out[dim] = M.mean(axis=1, skipna=True)
        cons = consistency(M, cols) if len(cols) > 1 else None
        info = {"columns": cols, "n_items": len(cols),
                "reversed": [c for c in cols if c in rev],
                "mean": float(out[dim].mean()) if out[dim].notna().any() else None,
                "alpha": (cons or {}).get("alpha"),
                "mean_r": (cons or {}).get("mean_r")}
        if cons and cons.get("alpha") is not None and cons["alpha"] < alpha_min:
            info["low_alpha"] = True
            audit["warnings"].append(
                "维度 %s 的内部一致性偏低（α=%.2f，%d 题）：合成分数含义模糊，"
                "结论要写清楚这一条" % (dim, cons["alpha"], len(cols)))
        audit["dimensions"][dim] = info
    return out, audit


def summarize(df: pd.DataFrame, target: str | None = None,
              roles: dict | None = None, items: list[str] | None = None,
              scale: tuple[int, int] = (SCALE_MIN, SCALE_MAX)) -> dict:
    roles = roles or detect_roles(df)
    target = target or pick_target(df, roles)
    num, cat = feature_split(df, target=target, roles=roles)
    items = items or num
    M = df[items].apply(pd.to_numeric, errors="coerce") if items else None
    out = {
        "n_respondents": int(len(df)),
        "n_items": len(items),
        "target": target,
        "categorical": cat,
        "id_columns": [c for c, r in roles.items() if r == "id"],
        "item_missing_rate": (float(M.isna().mean().mean()) if M is not None else None),
        "out_of_range": None,
        "straight_line": None,
    }
    if M is not None and len(M):
        lo, hi = float(scale[0]), float(scale[1])
        bad = int(((M < lo - 1e-9) | (M > hi + 1e-9)).any(axis=1).sum())
        out["out_of_range"] = bad
        out["straight_line"] = int(straight_lining(df, items).sum())
    return out


def trait_of(col: str) -> str:
    name = str(col)
    best, blen = None, 0
    for keys, trait in TRAITS:
        for k in keys:
            if k.lower() in name.lower() and len(k) > blen:
                best, blen = trait, len(k)
    return best or "未归类维度"


def explain(rows: list[dict], df: pd.DataFrame | None = None,
            cat_names: list[str] | None = None,
            trait_map: dict | None = None) -> list[dict]:
    cat_set = set(cat_names or [])
    tmap = trait_map or {}
    out = []
    for r in rows:
        name = str(r.get("feature", ""))
        w = float(r.get("weight", 0.0))
        z = float(r.get("z", 0.0))
        lvl = r.get("level_fdr", r.get("level", "red"))
        q = r.get("q", None)
        trait = ("分组变量" if name in cat_set
                 else tmap.get(name) or trait_of(name))
        if name in cat_set:
            tail = {"green": "该人群变量确实区分出不同的作答群体",
                    "amber": "区分方向可信但精度不足",
                    "red": "看不出该人群变量有区分作用"}.get(
                        lvl, "权重在边界上，等价于该变量未参与分型")
        else:
            if lvl == "green":
                tail = "该维度对分型的贡献可被数据支持"
            elif lvl == "amber":
                tail = "方向可信但精度不足"
            elif lvl == "red":
                tail = ("多重检验校正后不显著（q=%s），当前样本量下无法把它和"
                        "「各维度等权」区分开"
                        % ("%.3f" % q if isinstance(q, float) else "n/a"))
            else:
                tail = "权重落在边界上，等价于该维度未参与分型"
        share = ""
        if rows:
            tot = sum(abs(float(x.get("weight", 0.0))) for x in rows) or 1.0
            share = "占权重 %.0f%%，" % (abs(w) / tot * 100)
        out.append({"feature": name, "trait": trait,
                    "text": "%s（%s）权重 %.3f，%sz=%.2f：%s。"
                            % (name, trait, w, share, z, tail)})
    return out


def caveats(summ: dict, audit: dict | None = None) -> list[str]:
    out = []
    if summ.get("straight_line"):
        out.append("检出直线作答 %d 份（所有题选同一个选项）：这是无效问卷，"
                   "留着会凭空造出一个簇，建议剔除。" % summ["straight_line"])
    if summ.get("out_of_range"):
        out.append("有 %d 份问卷的作答超出计分范围：核对录入或问卷版本。"
                   % summ["out_of_range"])
    mr = summ.get("item_missing_rate")
    if isinstance(mr, float) and mr > 0.10:
        out.append("题目平均缺失率 %.0f%%：超过 10%%，先查是跳题逻辑还是录入问题。"
                   % (mr * 100))
    for w in (audit or {}).get("warnings", []):
        out.append(w)
    return out


def drop_invalid(df: pd.DataFrame, items: list[str], tol: int = 0
                 ) -> tuple[pd.DataFrame, int]:
    """剔除直线作答的问卷。返回 (剩余表, 剔除份数)。"""
    flag = straight_lining(df, items, tol=tol)
    return df[~flag].reset_index(drop=True), int(flag.sum())


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path) if str(path).lower().endswith(".csv") else pd.read_excel(path)
    df.columns = [re.sub(r"^﻿", "", str(c)).strip() for c in df.columns]
    return df
