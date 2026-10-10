# -*- coding: utf-8 -*-
"""数据体检与修补：真实表格的第一道关。

v1 的做法是 df.dropna() —— 缺一个格子就废掉一整行, 真实表里能砍掉一半样本。
本模块把体检和修补做成可复用、可报告、可序列化的对象:

    from hybridscope.prep import Prep
    p = Prep().fit(df, target="总评")
    df2 = p.transform(df)        # 同一套规则可用于新数据
    p.report()                   # 每一列被怎么处理、为什么, 人能读懂

修补策略(全部可关):
  * 标识列(学号/编号/姓名, 且几乎每行都不同) 不进特征
  * 缺失率 > max_missing 或取值唯一的列 直接剔除
  * 数值列中位数填补 + 分位截断(默认 1%/99%)
  * 类别列缺失独立成"缺失"档, 低频档并入"其他"
  * 重复行剔除
  * 各列量级相差 > scale_ratio 倍时自动标准化(加权距离对量纲敏感)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_HINTS = ("学号", "编号", "工号", "考号", "考生号", "身份证", "姓名", "序号",
            "id", "ID", "Id", "name", "code", "key", "uuid")
TARGET_HINTS = ("总评", "总分", "总成绩", "成绩", "得分", "score", "label",
                "目标", "档位", "等级", "target", "y")
CAT_HINTS = ("班", "专业", "学院", "性别", "学期", "年级", "类别", "类型",
             "地区", "城市", "科目", "课程", "档次", "分组")

MISSING_LEVEL = "缺失"
OTHER_LEVEL = "其他"


def _is_num(s: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(s))


def _looks_like_id(name: str) -> bool:
    low = str(name)
    return any(h in low for h in ID_HINTS)


def _looks_like_cat(name: str) -> bool:
    low = str(name)
    return any(h in low for h in CAT_HINTS)


def scan_column(s: pd.Series, name: str, force_cat: bool = False) -> dict:
    """单列体检：不还手，只报告。"""
    n = len(s)
    miss = float(s.isna().mean()) if n else 0.0
    nunique = int(s.nunique(dropna=True))
    out = {
        "column": str(name),
        "dtype": str(s.dtype),
        "n": int(n),
        "missing": round(miss, 4),
        "nunique": nunique,
        "unique_ratio": round(nunique / n, 4) if n else 0.0,
        "kind": "cat" if force_cat or not _is_num(s) else "num",
        "flags": [],
    }
    if nunique <= 1:
        out["flags"].append("constant")
    if miss > 0:
        out["flags"].append("missing")
    if out["unique_ratio"] > 0.9 and nunique > 20:
        out["flags"].append("id-like")
    if out["kind"] == "cat" and nunique > 50:
        out["flags"].append("high-cardinality")
    if out["kind"] == "num" and nunique > 2:
        v = pd.to_numeric(s, errors="coerce")
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = float(q3 - q1)
        if iqr > 0:
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            out["outlier"] = round(float(((v < lo) | (v > hi)).mean()), 4)
            if out["outlier"] > 0:
                out["flags"].append("outlier")
        out["sd"] = round(float(v.std()), 6) if v.notna().sum() > 1 else 0.0
    return out


def health_report(df: pd.DataFrame, target: str | None = None,
                  cat_cols: list[str] | None = None) -> dict:
    """整表体检：只报告不动手。"""
    cols = [c for c in df.columns if c != target]
    per = [scan_column(df[c], c, force_cat=bool(cat_cols and c in cat_cols))
           for c in cols]
    n = len(df)
    return {
        "n_rows": int(n),
        "n_cols": int(df.shape[1]),
        "target": target,
        "n_dup_rows": int(df.duplicated().sum()),
        "droppable_by_dropna": int(n - len(df.dropna())),
        "columns": per,
        "flags": {c["column"]: c["flags"] for c in per if c["flags"]},
    }


class Prep:
    """体检 + 修补。fit 记住规则，transform 可对任意新表复现同一套处理。"""

    def __init__(self, max_missing: float = 0.4, min_count: int = 3,
                 clip: tuple[float, float] | None = (0.01, 0.99),
                 drop_ids: bool = True, scale: str = "auto",
                 scale_ratio: float = 10.0, drop_dup_rows: str | bool = "auto",
                 discrete_max_levels: int = 25):
        """drop_dup_rows='auto' 时，先判断重复行是「重复录入」还是「多人同答」。

        量表/分档数据（每列取值就那么几个）里，重复行是正常的：十个人作答
        完全一样很常见。这类表若照搬「重复行=重复录入」的规则，会凭空丢掉
        一大半样本（实测 150 份问卷被剔到 97 份）。
        """
        self.max_missing = max_missing
        self.min_count = min_count
        self.clip = clip
        self.drop_ids = drop_ids
        self.scale = scale
        self.scale_ratio = scale_ratio
        self.drop_dup_rows = drop_dup_rows
        self.discrete_max_levels = discrete_max_levels

    # ---- fit ----
    def fit(self, df: pd.DataFrame, target: str | None = None,
            cat_cols: list[str] | None = None):
        self.target_ = target
        self.raw_columns_ = list(df.columns)
        self.scan_ = health_report(df, target=target, cat_cols=cat_cols)
        work = pd.DataFrame(df).copy()
        n0 = len(work)
        cols = [c for c in work.columns if c != target]
        # 标识列按名字排除（它天然把基数拉满）；连续列不排除——有连续列时
        # 重复行意味着「连测量值都一样」，那确实是重复录入，该剔。
        lv = [int(work[c].nunique(dropna=True)) for c in cols
              if not _looks_like_id(c)] or [0]
        self.discrete_ = bool(max(lv) <= int(self.discrete_max_levels))
        self.dup_rule_ = self.drop_dup_rows
        if self.drop_dup_rows == "auto":
            # 取值基数低 → 重复行是真实的多人同答，不是重复录入
            self.dup_rule_ = not self.discrete_
        if self.dup_rule_:
            work = work.drop_duplicates().reset_index(drop=True)
        self.n_dup_dropped_ = n0 - len(work)

        self.actions_ = []
        self.dropped_ = []
        self.num_fill_ = {}
        self.cat_map_ = {}
        self.clip_bounds_ = {}
        self.kept_ = []

        for c in self.raw_columns_:
            if c == target:
                continue
            s = work[c]
            force_cat = bool(cat_cols and c in cat_cols)
            kind = "cat" if (force_cat or not _is_num(s)) else "num"
            miss = float(s.isna().mean()) if len(s) else 1.0
            nunique = int(s.nunique(dropna=True))

            if self.drop_ids and _looks_like_id(c) and nunique > 20 \
                    and nunique / max(len(s), 1) > 0.9:
                self._drop(c, "标识列：取值几乎每行都不同，进距离只会制造噪声")
                continue
            if miss > self.max_missing:
                self._drop(c, "缺失率 %.0f%% 超过上限 %.0f%%" % (miss * 100, self.max_missing * 100))
                continue
            if nunique <= 1:
                self._drop(c, "取值唯一（常量列），不携带可分型信息")
                continue

            if kind == "num":
                v = pd.to_numeric(s, errors="coerce")
                if v.isna().all():
                    self._drop(c, "无法解析为数值，且未被声明为类别列")
                    continue
                med = float(v.median()) if v.notna().any() else 0.0
                self.num_fill_[c] = med
                if miss > 0:
                    self._act(c, "缺失 %.0f%% → 中位数 %.4g 填补" % (miss * 100, med))
                if self.clip is not None:
                    lo = float(v.quantile(self.clip[0]))
                    hi = float(v.quantile(self.clip[1]))
                    self.clip_bounds_[c] = (lo, hi)
                    n_out = int(((v < lo) | (v > hi)).sum())
                    if n_out:
                        self._act(c, "离群 %d 个 → 截断到 [%.4g, %.4g]" % (n_out, lo, hi))
                if _looks_like_cat(c) and nunique <= 20:
                    pass
            else:
                levels = s.astype(str).where(s.notna(), MISSING_LEVEL)
                vc = levels.value_counts()
                keep = [k for k, v in vc.items() if v >= self.min_count]
                if len(keep) < len(vc):
                    self._act(c, "%d 个低频档（< %d 例）并入「%s」"
                              % (len(vc) - len(keep), self.min_count, OTHER_LEVEL))
                if miss > 0:
                    self._act(c, "缺失 %.0f%% → 独立成「%s」档" % (miss * 100, MISSING_LEVEL))
                m = {k: (k if k in keep else OTHER_LEVEL) for k in vc.index}
                if OTHER_LEVEL not in set(m.values()) and len(keep) < len(vc):
                    pass
                self.cat_map_[c] = m
                if nunique > 50:
                    self._act(c, "高基数类别（%d 档）：距离里会退化成近乎全不相等，建议合并语义档" % nunique)
            self.kept_.append(c)

        # 量纲
        self.scale_ = False
        self.scale_mean_ = None
        self.scale_sd_ = None
        if self.scale == "auto":
            sds = []
            for c in self.kept_:
                if c in self.num_fill_:
                    v = pd.to_numeric(work[c], errors="coerce")
                    sd = float(v.std())
                    if sd > 0:
                        sds.append(sd)
            if len(sds) >= 2 and (max(sds) / max(min(sds), 1e-12)) > self.scale_ratio:
                self.scale_ = True
                self._act("(数值列)", "量级相差 %.0f 倍 > %.0f → 统一标准化"
                          % (max(sds) / max(min(sds), 1e-12), self.scale_ratio))
        elif self.scale == "on":
            self.scale_ = True
        if self.scale_:
            num_kept = [c for c in self.kept_ if c in self.num_fill_]
            M = work[num_kept].apply(pd.to_numeric, errors="coerce")
            self.scale_mean_ = M.mean().to_dict()
            self.scale_sd_ = M.std().replace(0, 1.0).to_dict()
        return self

    def _drop(self, col, reason):
        self.dropped_.append({"column": str(col), "reason": reason})

    def _act(self, col, reason):
        self.actions_.append({"column": str(col), "action": reason})

    # ---- transform ----
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(df).copy()
        for c in [d["column"] for d in self.dropped_]:
            if c in out.columns:
                out = out.drop(columns=[c])

        for c, med in self.num_fill_.items():
            if c not in out.columns:
                out[c] = np.nan
            v = pd.to_numeric(out[c], errors="coerce")
            v = v.fillna(med)
            if c in self.clip_bounds_:
                lo, hi = self.clip_bounds_[c]
                v = v.clip(lo, hi)
            out[c] = v

        for c, m in self.cat_map_.items():
            if c not in out.columns:
                out[c] = MISSING_LEVEL
            lv = out[c].astype(str).where(out[c].notna(), MISSING_LEVEL)
            out[c] = lv.map(lambda x: m.get(x, OTHER_LEVEL))

        if self.scale_ and self.scale_mean_:
            for c, mu in self.scale_mean_.items():
                if c in out.columns:
                    sd = self.scale_sd_.get(c, 1.0) or 1.0
                    out[c] = (pd.to_numeric(out[c], errors="coerce") - mu) / sd

        cols = [c for c in self.kept_ if c in out.columns]
        if self.target_ is not None and self.target_ in out.columns:
            cols = cols + [self.target_]
        out = out[cols]
        if self.target_ is not None and self.target_ in out.columns:
            out = out[out[self.target_].notna()].reset_index(drop=True)
        return out.reset_index(drop=True)

    def fit_transform(self, df, target=None, cat_cols=None) -> pd.DataFrame:
        """fit 后按同一套规则转换；重复行在 fit 里已统计，这里同步剔除。"""
        self.fit(df, target=target, cat_cols=cat_cols)
        work = pd.DataFrame(df)
        if self.dup_rule_:
            work = work.drop_duplicates()
        return self.transform(work)

    # ---- report ----
    def report(self) -> dict:
        return {
            "n_rows_in": int(self.scan_["n_rows"]),
            "n_dup_rows_dropped": int(self.n_dup_dropped_),
            "n_dropped_cols": len(self.dropped_),
            "dropped": self.dropped_,
            "actions": self.actions_,
            "kept": [str(c) for c in self.kept_],
            "scaled": bool(self.scale_),
            "dropna_would_drop": int(self.scan_["droppable_by_dropna"]),
            "flags": self.scan_["flags"],
            "discrete_table": bool(getattr(self, "discrete_", False)),
        }

    def to_text(self) -> str:
        r = self.report()
        lines = ["数据体检：%d 行 / %d 列，重复行 %d 条%s"
                 % (r["n_rows_in"], self.scan_["n_cols"], r["n_dup_rows_dropped"],
                    "（取值基数低，判定为多人同答，未剔除）"
                    if r.get("discrete_table") and not r["n_dup_rows_dropped"] else "")]
        if r["dropna_would_drop"]:
            lines.append("（若沿用整行丢弃缺失的做法，会损失 %d 行）" % r["dropna_would_drop"])
        if r["dropped"]:
            lines.append("剔除列：")
            lines += ["  - %s：%s" % (d["column"], d["reason"]) for d in r["dropped"]]
        if r["actions"]:
            lines.append("修补动作：")
            lines += ["  - %s：%s" % (a["column"], a["action"]) for a in r["actions"]]
        if not r["actions"] and not r["dropped"]:
            lines.append("未发现需要处理的问题。")
        return "\n".join(lines)


def prepare(df, target=None, cat_cols=None, **kw) -> tuple[pd.DataFrame, Prep]:
    """一步到位：返回 (清洗后的表, Prep 对象)。"""
    p = Prep(**kw)
    return p.fit_transform(df, target=target, cat_cols=cat_cols), p
