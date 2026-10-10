# -*- coding: utf-8 -*-
"""字段映射层：把不同批次/不同来源的表对齐到同一套 canonical 字段。

跨批次对比的失败，绝大多数不是算法问题，而是「这一批叫 A、下一批叫 B」。
同一门课换个学期，教务系统给的列名、分项个数、评分口径都会变：上学期是
「平时作业 / 大作业」两项，下学期拆成「测验 1–4 / 报告 1–4」八项。直接把
两份表喂给同一个模型，得到的权重差异里混着「口径变了」和「真的变了」，
两者无法从结果里分离。

本模块做的事很朴素：先把「什么叫同一个东西」显式写出来（Schema），再把每一批
原始列按规则折叠到 canonical 字段（apply），并在折叠前先检查这批列能不能
折（audit）。折叠不是免费的：把四次测验平均成一个「过程考核」得分，前提是这
四次测验量纲一致、且测的是相近的东西。若内部一致性很低，合并会掩盖差异，
这时模块给出警告，而不是默默平均。

三个层次要分清：
  1) 原始层：各批次自己的列，学期内分析用它，结论只对这一批成立；
  2) canonical 层：跨批次可比的字段，跨批次对比只在这一层做；
  3) 不可比层：语义不同但同名的列（比如「专业」这学期填专业名、下学期填班级名），
     显式标 comparable=False，对比时排除，避免拿 Δw 讲没有含义的故事。

Schema 可以 JSON 序列化，映射规则因此是可存档、可审阅、可复用的资产，
而不是散落在脚本里的 if-else。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

NUMS = ("mean", "wmean", "sum", "first", "max", "min")
CATS = ("first", "concat", "majority")
ROLES = ("feature", "target", "id", "exclude")


# ---------------------------------------------------------------- 字段定义

@dataclass
class FieldSpec:
    """一个 canonical 字段。

    name      canonical 名，跨批次共用（如 process / project / total）
    label     中文名，报告里给人看
    kind      numeric / categorical
    agg       多列折叠方式：numeric 用 mean/wmean/sum/first/max/min；
              categorical 用 first/concat/majority
    weights   wmean 时的列权重（不写则等权）
    scale     折叠后乘的常数（如把百分制换算成「对总评的贡献」= 0.6×）
    role      feature=进模型 / target=只用来切标签 / id=标识 / exclude=只存档
    required  缺失时是否算失败
    comparable 是否允许跨批次比较（同名不同义的列必须写 False）
    note      写给人看的口径说明
    """
    name: str
    label: str = ""
    kind: str = "numeric"
    agg: str = "mean"
    weights: dict | None = None
    scale: float = 1.0
    role: str = "feature"
    required: bool = True
    comparable: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v not in (None, "", 1.0) or k in ("name", "agg", "kind", "role")}

    @staticmethod
    def from_dict(d: dict) -> "FieldSpec":
        known = {f for f in FieldSpec.__dataclass_fields__}
        return FieldSpec(**{k: v for k, v in d.items() if k in known})


@dataclass
class Unified:
    """一次 apply 的结果：对齐后的表 + 审计报告。"""
    df: pd.DataFrame
    profile: str
    report: dict

    @property
    def warnings(self) -> list[str]:
        return self.report.get("warnings", [])

    def ok(self) -> bool:
        return not self.report.get("errors")


# ---------------------------------------------------------------- 折叠前体检

def _numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def consistency(M: pd.DataFrame, cols: list[str]) -> dict:
    """一组数值列能不能合成一个字段：量纲是否一致、测的是不是相近的东西。

    只做两件有把握的判断：
      · 量纲：各列标准差的比值，差一个数量级就不该直接平均；
      · 内部一致性：Cronbach α（标准化后），α 低说明这几列不是同一构念的
        重复测量，平均出来的分数含义模糊。
    """
    sub = M[cols].astype(float)
    k = len(cols)
    info = {"n_cols": k, "columns": list(cols), "alpha": None,
            "mean_r": None, "sd_ratio": None, "p95_ratio": None,
            "missing_rate": float(sub.isna().mean().mean()),
            "level": "ok", "messages": []}
    if k < 2:
        return info

    sd = sub.std(ddof=1, skipna=True)
    if (sd > 0).all():
        info["sd_ratio"] = float(sd.max() / max(sd.min(), 1e-12))
    p95 = sub.quantile(0.95)
    p95 = p95.where(p95.abs() > 1e-12, np.nan)
    if p95.notna().all():
        info["p95_ratio"] = float(p95.max() / max(p95.min(), 1e-12))

    C = sub.corr(min_periods=3)
    vals = []
    for i in range(k):
        for j in range(i + 1, k):
            v = C.iloc[i, j]
            if np.isfinite(v):
                vals.append(float(v))
    if vals:
        r = float(np.mean(vals))
        info["mean_r"] = r
        # Cronbach α（标准化项）：α = k·r̄ / (1 + (k−1)·r̄)
        denom = 1.0 + (k - 1) * r
        info["alpha"] = float(k * r / denom) if denom > 1e-12 else None
    a = info["alpha"]
    if a is not None and a < 0.5:
        info["level"] = "warn"
        info["messages"].append(
            "内部一致性低（Cronbach α=%.2f，平均相关 %.2f）：这几项测的不像同一个东西，"
            "平均后会掩盖分项差异。跨批次结论请用合并后的字段，学期内结论请回到分项。"
            % (a, info["mean_r"]))
    elif a is not None and a < 0.7:
        info["messages"].append(
            "内部一致性中等（α=%.2f）：可以合并，但合并分数里掺着分项各自的噪声。"
            % a)
    if info["sd_ratio"] and info["sd_ratio"] > 3.0:
        info["level"] = "warn"
        info["messages"].append(
            "量纲不一致（标准差相差 %.1f 倍）：直接平均会让大方差项主导，建议先标准化。"
            % info["sd_ratio"])
    if info["p95_ratio"] and info["p95_ratio"] > 3.0:
        info["level"] = "warn"
        info["messages"].append(
            "取值上界相差 %.1f 倍：两列可能不是同一个计分尺度。" % info["p95_ratio"])
    return info


def _collapse_num(M: pd.DataFrame, cols: list[str], agg: str,
                  weights: dict | None, scale: float) -> pd.Series:
    sub = M[cols].astype(float)
    if agg == "first":
        v = sub[cols[0]]
    elif agg == "sum":
        v = sub.sum(axis=1, min_count=1)
    elif agg == "max":
        v = sub.max(axis=1, skipna=True)
    elif agg == "min":
        v = sub.min(axis=1, skipna=True)
    elif agg == "wmean":
        w = np.array([float((weights or {}).get(c, 1.0)) for c in cols], float)
        if w.sum() <= 0:
            w = np.ones(len(cols), float)
        W = pd.DataFrame(np.tile(w, (len(sub), 1)), index=sub.index, columns=cols)
        avail = sub.notna()
        W = W.where(avail, 0.0)
        denom = W.sum(axis=1)
        num = (sub.fillna(0.0) * W).sum(axis=1)
        v = num / denom.where(denom > 1e-12, np.nan)
    else:                                   # mean
        v = sub.mean(axis=1, skipna=True)
    return v * float(scale)


def _collapse_cat(df: pd.DataFrame, cols: list[str], agg: str) -> pd.Series:
    S = df[cols].astype("object")
    if agg == "concat":
        return S.fillna("").astype(str).agg("|".join, axis=1).replace("", np.nan)
    if agg == "majority":
        def pick(row):
            vals = [x for x in row if isinstance(x, str) and x not in ("", "nan")]
            if not vals:
                return np.nan
            return pd.Series(vals).value_counts().index[0]
        return S.apply(pick, axis=1)
    return S[cols[0]]


# ---------------------------------------------------------------- Schema

class Schema:
    """canonical 字段表 + 各批次到 canonical 的映射。

    profiles 形如 {"2026spring": {"process": ["测验1","测验2","测验3","测验4"],
                                   "project": ["报告1", ...], "total": ["总评"]}, ...}
    值可以是列名列表，也可以是 {"columns": [...], "agg": "wmean", "weights": {...}}。
    """

    def __init__(self, name: str, fields: list[FieldSpec],
                 profiles: dict[str, dict], version: str = "1",
                 note: str = ""):
        self.name = name
        self.version = str(version)
        self.note = note
        self.fields = list(fields)
        self.profiles = {k: dict(v) for k, v in profiles.items()}
        self._by_name = {f.name: f for f in self.fields}

    # -- 构造 / 序列化 ------------------------------------------------
    def to_dict(self) -> dict:
        return {"name": self.name, "version": self.version, "note": self.note,
                "fields": [f.to_dict() for f in self.fields],
                "profiles": self.profiles}

    @staticmethod
    def from_dict(d: dict) -> "Schema":
        return Schema(
            name=d.get("name", "schema"),
            fields=[FieldSpec.from_dict(x) for x in d.get("fields", [])],
            profiles={k: v for k, v in d.get("profiles", {}).items()},
            version=str(d.get("version", "1")),
            note=d.get("note", ""),
        )

    def to_json(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @staticmethod
    def from_json(path: str) -> "Schema":
        with open(path, "r", encoding="utf-8") as f:
            return Schema.from_dict(json.load(f))

    def __repr__(self) -> str:  # pragma: no cover
        return "Schema(%s v%s, %d fields, profiles=%s)" % (
            self.name, self.version, len(self.fields), list(self.profiles))

    # -- 查询 --------------------------------------------------------
    @property
    def canonical_names(self) -> list[str]:
        return [f.name for f in self.fields]

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.fields if f.role == "feature"]

    @property
    def comparable_names(self) -> list[str]:
        return [f.name for f in self.fields if f.role == "feature" and f.comparable]

    def spec(self, name: str) -> FieldSpec | None:
        return self._by_name.get(name)

    def sources(self, profile: str, name: str) -> list[str]:
        """取该批次该字段的源列名；未知批次/字段返回空列表。"""
        p = self.profiles.get(profile) or {}
        v = p.get(name)
        if v is None:
            return []
        if isinstance(v, dict):
            cols = v.get("columns") or []
            return [str(c) for c in cols]
        return [str(c) for c in v]

    def missing_columns(self, df: pd.DataFrame, profile: str) -> list[str]:
        return [c for f in self.fields for c in self.sources(profile, f.name)
                if c not in df.columns]

    # -- 应用 --------------------------------------------------------
    def apply(self, df: pd.DataFrame, profile: str,
              extra: dict | None = None) -> Unified:
        """把一批原始表折叠成 canonical 表。"""
        if profile not in self.profiles:
            raise KeyError("未知批次 %r；已有：%s" % (profile, list(self.profiles)))
        errors: list[str] = []
        warnings: list[str] = []
        items: list[dict] = []
        out = pd.DataFrame(index=df.index)
        used: set[str] = set()

        for f in self.fields:
            cols = [c for c in self.sources(profile, f.name) if c in df.columns]
            absent = [c for c in self.sources(profile, f.name) if c not in df.columns]
            if not cols:
                msg = "批次 %s 缺少字段 %s（%s）：源列 %s 不存在"
                if f.required:
                    errors.append(msg % (profile, f.name, f.label, absent or "未配置"))
                    items.append({"field": f.name, "label": f.label, "sources": [],
                                  "status": "missing"})
                else:
                    warnings.append(msg % (profile, f.name, f.label, absent or "未配置"))
                    items.append({"field": f.name, "label": f.label, "sources": [],
                                  "status": "absent"})
                continue
            if absent:
                warnings.append("字段 %s 的源列 %s 在本批次不存在，已跳过"
                                % (f.name, "、".join(absent)))

            cfg = self.profiles[profile].get(f.name)
            if isinstance(cfg, dict):
                agg = str(cfg.get("agg", f.agg))
                weights = cfg.get("weights", f.weights)
                scale = float(cfg.get("scale", f.scale))
                zero_nan = bool(cfg.get("zero_as_missing", False))
            else:
                agg, weights, scale = f.agg, f.weights, f.scale
                zero_nan = False

            if f.kind == "categorical":
                ser = _collapse_cat(df, cols, agg if agg in CATS else "first")
                cons = None
            else:
                M = _numeric_frame(df, cols)
                if zero_nan:                # 「0 分」其实是「未交/缺考」，不是成绩
                    M = M.mask(M == 0)
                if len(cols) > 1:
                    cons = consistency(M, cols)
                    for m in cons["messages"]:
                        warnings.append("[%s] %s" % (f.name, m))
                else:
                    cons = None
                ser = _collapse_num(M, cols, agg if agg in NUMS else "mean",
                                    weights, scale)
            out[f.name] = ser
            used.update(cols)
            stats = {}
            if f.kind == "numeric":     # 只对数值字段算分布量，学号等标识列别去碰
                num_ser = pd.to_numeric(ser, errors="coerce")
                n_ok = int(num_ser.notna().sum())
                if n_ok:
                    stats = {
                        "missing_out": int(len(ser) - num_ser.notna().sum()),
                        "mean": float(num_ser.mean()),
                        "sd": float(num_ser.std(ddof=1)) if n_ok > 1 else None,
                        "zero_rate": float((num_ser == 0).mean()),
                    }
                else:
                    stats = {"missing_out": int(len(ser)), "mean": None,
                             "sd": None, "zero_rate": None}
            else:
                stats = {"missing_out": int(ser.isna().sum())}
            items.append({
                "field": f.name, "label": f.label, "kind": f.kind,
                "sources": cols, "agg": agg, "scale": scale,
                "role": f.role, "comparable": f.comparable,
                "consistency": cons, "status": "ok", **stats,
            })

        unused = [str(c) for c in df.columns if c not in used]
        for c in unused:
            warnings.append("列 %s 未纳入任何 canonical 字段，本口径下被丢弃" % c)
        if extra:
            for k, v in extra.items():
                out[k] = v

        rep = {
            "schema": self.name, "version": self.version, "profile": profile,
            "n_rows": int(len(out)), "fields": items,
            "unused_columns": unused, "warnings": warnings, "errors": errors,
            "comparable": [f.name for f in self.fields if f.comparable],
        }
        return Unified(df=out, profile=profile, report=rep)

    # -- 跨批次可比性 ------------------------------------------------
    def compatibility(self, frames: dict[str, pd.DataFrame],
                      scale_tol: float = 3.0,
                      miss_tol: float = 0.20) -> dict:
        """给定各批次原始表，判断 canonical 层哪些字段真的能跨批次比。

        判定用三件事：字段在两边都存在；FieldSpec 声明可比较；两端的取值尺度
        与缺失率没有差到「不是同一个量」。注意这里只查尺度，不查均值——
        均值变了那是要检验的漂移，不是不可比。
        """
        unis = {p: self.apply(df, p) for p, df in frames.items()}
        names = self.canonical_names
        rows = []
        for f in self.fields:
            srcs = {p: self.sources(p, f.name) for p in frames}
            have = [p for p in frames if srcs.get(p) and f.name in unis[p].df.columns]
            row = {"field": f.name, "label": f.label, "role": f.role,
                   "comparable": f.comparable,
                   "sources": {p: srcs.get(p, []) for p in frames},
                   "present_in": have, "reasons": [], "cautions": [], "stats": {}}
            if f.role != "feature":
                row["comparable"] = False
                row["reasons"].append("该字段是 %s，不参与权重比较" % f.role)
                rows.append(row)
                continue
            if len(have) < 2:
                row["comparable"] = False
                row["reasons"].append("只在 %s 有该字段，另一批次没有" % "、".join(have))
                rows.append(row)
                continue
            if not f.comparable:
                row["reasons"].append("声明为不可比（同名不同义 / 语义随批次变）")
            if f.kind == "numeric":
                sds, p95s, miss = {}, {}, {}
                for p in have:
                    s = pd.to_numeric(unis[p].df[f.name], errors="coerce")
                    sds[p] = float(s.std(ddof=1)) if s.notna().sum() > 2 else np.nan
                    p95s[p] = float(s.quantile(0.95)) if s.notna().sum() else np.nan
                    miss[p] = float(s.isna().mean())
                row["stats"] = {"std": sds, "p95": p95s, "missing_rate": miss}
                vals = [v for v in p95s.values() if np.isfinite(v) and abs(v) > 1e-12]
                if len(vals) >= 2:
                    ratio = max(abs(v) for v in vals) / max(min(abs(v) for v in vals), 1e-12)
                    if ratio > scale_tol:
                        row["comparable"] = False
                        row["reasons"].append(
                            "取值尺度相差 %.1f 倍（%s），不是同一个计分量纲"
                            % (ratio, " vs ".join("%.3g" % v for v in vals)))
                dm = max(miss.values()) - min(miss.values())
                if dm > miss_tol:
                    row["comparable"] = False
                    row["reasons"].append(
                        "缺失率相差 %.0f 个百分点：一边的缺失可能是口径不同（如缺考记 0 还是留空）"
                        % (dm * 100))
                # 单侧常数列：一端没有变异，Δw 无从解释
                for p in have:
                    if np.isfinite(sds[p]) and sds[p] < 1e-12:
                        row["comparable"] = False
                        row["reasons"].append("%s 的该字段是常数列，没有变异可比" % p)
                zr = {}
                for p in have:
                    it = next((x for x in unis[p].report["fields"]
                               if x["field"] == f.name), None)
                    if it and it.get("zero_rate") is not None:
                        zr[p] = it["zero_rate"]
                if len(zr) >= 2 and (max(zr.values()) - min(zr.values())) > 0.10:
                    row["cautions"].append(
                        "零分比例相差 %.0f 个百分点（%s）：一边的「未交」可能记 0、"
                        "另一边留空或按缺考处理，均值差里掺着这个口径差"
                        % ((max(zr.values()) - min(zr.values())) * 100,
                           "、".join("%s %.0f%%" % (p, v * 100) for p, v in zr.items())))
            else:
                levels = {}
                for p in have:
                    s = unis[p].df[f.name].astype(str)
                    levels[p] = sorted(set(s.dropna().tolist()))
                row["stats"] = {"levels": levels}
                keys = list(levels.values())
                if keys:
                    inter = set(keys[0])
                    for k in keys[1:]:
                        inter &= set(k)
                    union = set().union(*keys)
                    ov = len(inter) / max(len(union), 1)
                    row["stats"]["overlap"] = ov
                    if ov < 0.10:
                        row["comparable"] = False
                        row["reasons"].append(
                            "取值域几乎不重叠（重合 %.2f）：%s 与 %s 填的不是同一种分组"
                            % (ov, " / ".join(levels[have[0]][:3]),
                               " / ".join(levels[have[1]][:3])))
            rows.append(row)

        ok = [r["field"] for r in rows if r["comparable"]]
        no = [r["field"] for r in rows
              if not r["comparable"] and r["role"] == "feature"]
        cautions = ["[%s] %s" % (r["field"], c) for r in rows for c in r["cautions"]]
        return {
            "schema": self.name, "version": self.version,
            "profiles": sorted(frames), "rows": rows,
            "comparable": ok, "not_comparable": no, "cautions": cautions,
            "verdict": ("%d 个字段可跨批次比较：%s" % (len(ok), "、".join(ok))
                        if ok else "没有字段能跨批次比较：先把字段口径对齐"),
        }

    # -- 建议（初稿） ------------------------------------------------
    @staticmethod
    def suggest(frames: dict[str, pd.DataFrame], target_hint: str | None = None
                ) -> dict[str, dict[str, list[str]]]:
        """给一份映射初稿：按列名前缀/数字尾号把同源分组，跨批次按名字匹配。

        只是初稿：它按字面相似分组，不保证语义正确，必须人工确认后落盘。
        """
        base: dict[str, dict] = {}
        for p, df in frames.items():
            groups: dict[str, list[str]] = {}
            for c in df.columns:
                s = str(c).strip()
                key = "".join(ch for ch in s if not ch.isdigit()).strip() or s
                if pd.api.types.is_numeric_dtype(df[c]) and key == s and s.isdigit():
                    key = "col"
                groups.setdefault(key, []).append(s)
            base[p] = {k: v for k, v in groups.items()}
        # 跨批次对齐：名字相同的组直接对齐；不同名的按组内列数 + 数值尺度配对
        keys = {p: set(g) for p, g in base.items()}
        common = set.intersection(*keys.values()) if keys else set()
        out: dict[str, dict[str, list[str]]] = {}
        for p, g in base.items():
            out[p] = {k: v for k, v in g.items() if k in common}
        return out


# ---------------------------------------------------------------- 可读报告

def audit_text(uni: Unified) -> str:
    r = uni.report
    L = ["字段映射审计：%s（schema %s v%s，%d 行）"
         % (uni.profile, r["schema"], r["version"], r["n_rows"])]
    for it in r["fields"]:
        if it["status"] != "ok":
            L.append("  %-12s %-10s 缺失（源列 %s）"
                     % (it["field"], it.get("label", ""), it.get("sources") or "未配置"))
            continue
        src = "、".join(it["sources"])
        tail = ""
        if len(it["sources"]) > 1:
            tail = " ← %s(%d 列)" % (it["agg"], len(it["sources"]))
        L.append("  %-12s %-10s ← %s%s" % (it["field"], it.get("label", ""), src, tail))
        c = it.get("consistency")
        if c and c.get("alpha") is not None:
            L.append("                α=%.2f  平均相关 %.2f  sd 比 %.1f"
                     % (c["alpha"], c["mean_r"], c.get("sd_ratio") or float("nan")))
    if r["unused_columns"]:
        L.append("  未纳入的列：%s" % "、".join(r["unused_columns"]))
    for w in r["warnings"]:
        L.append("  [警告] %s" % w)
    for e in r["errors"]:
        L.append("  [错误] %s" % e)
    return "\n".join(L)


def compat_text(cp: dict) -> str:
    L = ["跨批次可比性（schema %s v%s，批次 %s）"
         % (cp["schema"], cp["version"], "、".join(cp["profiles"]))]
    for r in cp["rows"]:
        mark = "可比" if r["comparable"] else "不可比"
        src = " | ".join("%s:%s" % (p, "、".join(c) or "—")
                         for p, c in r["sources"].items())
        L.append("  [%-4s] %-12s %s" % (mark, r["field"], src))
        for why in r["reasons"]:
            L.append("            原因：%s" % why)
        for c in r.get("cautions", []):
            L.append("            提示：%s" % c)
    for c in cp.get("cautions", []):
        if not any(c in r.get("cautions", []) for r in cp["rows"]):
            L.append("  [提示] %s" % c)
    L.append("  " + cp["verdict"])
    return "\n".join(L)


def harmonize(schema: Schema, frames: dict[str, pd.DataFrame],
              term_col: str = "term") -> tuple[pd.DataFrame, dict]:
    """把多批数据对齐后纵向合并成一张长表，附带可比性结论。

    长表比两个分文件好用：一次建模就能带上「批次」这个分组变量，
    也便于把两批画在同一张图里。
    """
    unis = {p: schema.apply(df, p, extra={term_col: p}) for p, df in frames.items()}
    cp = schema.compatibility(frames)
    long = pd.concat([u.df for u in unis.values()], ignore_index=True)
    meta = {"schema": schema.to_dict(), "compatibility": cp,
            "audits": {p: u.report for p, u in unis.items()}}
    return long, meta
