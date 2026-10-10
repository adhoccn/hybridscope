# -*- coding: utf-8 -*-
"""表格接入：自动区分数值列与类别列，输出算法可直接吃的矩阵。"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def split_columns(df: pd.DataFrame, target: str | None = None,
                  cat_cols: list[str] | None = None):
    """返回 X_num, X_cat, num_names, cat_names, y。"""
    df = df.dropna()
    if target is not None and target in df.columns:
        y = df[target].astype("category").cat.codes.to_numpy()
        X = df.drop(columns=[target])
    else:
        y = None
        X = df

    if cat_cols is None:
        cat_names = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    else:
        cat_names = [c for c in cat_cols if c in X.columns]
    X = X.copy()
    num_names = [c for c in X.columns if c not in cat_names]

    # 未声明为类别、又不是数值型的列：能转成数字就转（缺值填中位数），
    # 转不动的说明它其实是类别列，按类别处理而不是让 to_numpy 抛异常。
    for c in list(num_names):
        v = pd.to_numeric(X[c], errors="coerce")
        if v.isna().all():
            num_names.remove(c)
            cat_names.append(c)
        else:
            X[c] = v.fillna(v.median() if v.notna().any() else 0.0)

    X_num = X[num_names].to_numpy(dtype=float) if num_names else np.zeros((len(X), 0))
    if cat_names:
        X_cat = np.column_stack([
            X[c].astype("category").cat.codes.to_numpy() for c in cat_names
        ])
    else:
        X_cat = np.zeros((len(X), 0), dtype=int)
    return X_num, X_cat, num_names, cat_names, y


def data_fingerprint(df: pd.DataFrame, target: str | None = None,
                     cat_cols: list[str] | None = None) -> dict:
    """数据指纹：回答「两次分析跑的是不是同一批数据」。

    hash_rows 对行序敏感（行序本身就是这条管线的建模对象）；
    hash_pool 对行序不敏感，用来区分「换了人」和「同一批人换了顺序」。
    """
    df = df.reset_index(drop=True)
    cols = [c for c in df.columns if c != target]
    if cat_cols is None:
        cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    cat_set = set(cat_cols)

    parts = []
    numeric, categorical = {}, {}
    for c in cols:
        s = df[c]
        if c in cat_set:
            vc = s.astype(str).value_counts()
            categorical[c] = {str(k): int(v) for k, v in vc.head(50).items()}
            parts.append("C|%s|%s" % (c, "|".join(
                "%s:%d" % kv for kv in categorical[c].items())))
            continue
        v = pd.to_numeric(s, errors="coerce")
        finite = v.replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            numeric[c] = {"n_missing": int(v.isna().sum()), "mean": None,
                          "std": None, "min": None, "max": None,
                          "q05": None, "q25": None, "q50": None,
                          "q75": None, "q95": None}
            parts.append("N|%s|empty" % c)
            continue
        qs = np.nanquantile(finite.to_numpy(float), [0.05, 0.25, 0.5, 0.75, 0.95])
        numeric[c] = {
            "n_missing": int(v.isna().sum()),
            "mean": float(finite.mean()), "std": float(finite.std(ddof=1))
            if len(finite) > 1 else 0.0,
            "min": float(finite.min()), "max": float(finite.max()),
            "q05": float(qs[0]), "q25": float(qs[1]), "q50": float(qs[2]),
            "q75": float(qs[3]), "q95": float(qs[4]),
        }
        # 数值指纹用 6 位小数，避免浮点末位抖动导致「同一批数据」被判成不同
        parts.append("N|%s|%s" % (c, ",".join(
            "%.6f" % x for x in finite.to_numpy(float))))

    row_str = "\n".join(parts)
    # 行序不敏感：把每列的值排序后再拼
    pool_parts = []
    for c in cols:
        s = df[c]
        if c in cat_set:
            # 统一转成 str 再排序：类别列里混着 NaN 时直接 sorted 会
            # 拿 str 和 float 比大小而抛 TypeError（成绩表里缺班级名就会撞上）
            pool_parts.append("C|" + ",".join(sorted(str(x) for x in s.tolist())))
        else:
            v = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
            pool_parts.append("N|" + ",".join("%.6f" % x for x in np.sort(v)))

    return {
        "n_rows": int(len(df)),
        "n_cols": int(len(cols)),
        "columns": [str(c) for c in cols],
        "cat_columns": [str(c) for c in cols if c in cat_set],
        "hash_rows": hashlib.sha256(row_str.encode("utf-8")).hexdigest()[:16],
        "hash_pool": hashlib.sha256(
            "\n".join(pool_parts).encode("utf-8")).hexdigest()[:16],
        "numeric": numeric,
        "categorical": categorical,
    }


def fingerprint_equal(a: dict | None, b: dict | None, pool: bool = False) -> bool:
    """两份指纹是否指向同一批数据。pool=True 时忽略行序。"""
    if not a or not b:
        return False
    key = "hash_pool" if pool else "hash_rows"
    return a.get(key) is not None and a.get(key) == b.get(key)


def load_table(path: str, target: str | None = None,
               cat_cols: list[str] | None = None):
    """读 CSV/Excel，返回与 split_columns 相同的五元组。"""
    if str(path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return split_columns(df, target=target, cat_cols=cat_cols)


def table_overview(frames: dict[str, pd.DataFrame]) -> dict:
    """多表字段对照：共有列、部分列、同名但类型冲突的列。"""
    names = list(frames)
    per = {n: list(frames[n].columns) for n in names}
    allcols: list[str] = []
    for n in names:
        for c in per[n]:
            if c not in allcols:
                allcols.append(c)
    common = [c for c in allcols if all(c in per[n] for n in names)]
    conflicts = []
    for c in common:
        kinds = set()
        for n in names:
            s = frames[n][c]
            kinds.add("num" if pd.api.types.is_numeric_dtype(s) else "cat")
        if len(kinds) > 1:
            conflicts.append(c)
    return {
        "names": names,
        "columns": per,
        "all": allcols,
        "common": common,
        "partial": [c for c in allcols if c not in common],
        "conflicts": conflicts,
        "n_rows": {n: int(len(frames[n])) for n in names},
    }


def concat_frames(frames: dict[str, pd.DataFrame], keep="intersection",
                  add_source=True, source_col="数据来源"):
    """纵向合并：多个文件是同一批字段的不同批次样本，拼行。

    keep=intersection 只留共有列（安全）；keep=union 保留全部列，
    缺失填 NaN，缺得多会在报告里标出来。
    """
    ov = table_overview(frames)
    cols = ov["common"] if keep == "intersection" else ov["all"]
    parts, filled = [], {}
    for n in ov["names"]:
        d = frames[n].copy()
        miss = [c for c in cols if c not in d.columns]
        for c in miss:
            d[c] = np.nan
        if miss:
            filled[n] = miss
        for c in ov["conflicts"]:
            if c in d.columns:
                d[c] = d[c].astype(str)
        d = d[cols]
        if add_source:
            d[source_col] = n
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    report = {
        "mode": "concat",
        "keep": keep,
        "files": ov["names"],
        "n_rows": ov["n_rows"],
        "n_total": int(len(df)),
        "kept_cols": cols,
        "dropped_cols": [c for c in ov["all"] if c not in cols],
        "filled": filled,
        "conflicts": ov["conflicts"],
        "source_col": source_col if add_source else None,
    }
    return df, report


def merge_frames(frames: dict[str, pd.DataFrame], key: str, how="inner"):
    """横向合并：多个文件是同一样本的不同字段，按主键对齐。

    同名非主键列自动加文件后缀，避免静默覆盖。
    """
    names = list(frames)
    key_stats = {}
    for n in names:
        s = frames[n][key]
        key_stats[n] = {
            "rows": int(len(frames[n])),
            "unique": int(s.nunique(dropna=True)),
            "dup": int(s.duplicated().sum()),
            "nan": int(s.isna().sum()),
        }
    df = frames[names[0]].copy()
    steps = []
    for n in names[1:]:
        before = len(df)
        df = df.merge(frames[n], on=key, how=how, suffixes=("", f"@{n}"))
        steps.append({"file": n, "before": before, "after": int(len(df))})
    report = {
        "mode": "merge",
        "key": key,
        "how": how,
        "files": names,
        "key_stats": key_stats,
        "steps": steps,
        "n_total": int(len(df)),
        "n_cols": int(df.shape[1]),
    }
    return df, report


def standardize(X_num: np.ndarray) -> np.ndarray:
    """按列标准化，零方差列保持为 0。"""
    X = np.asarray(X_num, float)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (X - X.mean(axis=0)) / sd
