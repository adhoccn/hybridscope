# -*- coding: utf-8 -*-
"""HTTP 服务：把 HybridScope 变成别的系统能调的一个接口。

命令行与工作台给人用，接口给系统用。场景很具体：教务系统每学期导出一张成绩
表，希望「上传→拿到分型结论与可信度」这一步自动化，而不是有人手动打开页面
点一遍；做对比时也希望直接把两份快照丢进来，拿回一句「真漂移还是噪声」。

启动：
  python -m hybridscope.serve --host 127.0.0.1 --port 8000
  uvicorn hybridscope.serve:app --reload

接口：
  GET  /health              存活与版本
  POST /analyze             上传 CSV/Excel → 完整分析（权重、标准误、FDR、可信度）
  POST /analyze/text        CSV 文本体 → 同上（便于程序直接 POST）
  POST /snapshot            上传 → 只返回快照（存档用，供以后 /compare）
  POST /compare             两份快照 → 漂移结论
  POST /unify               多张表 + 字段映射 → 对齐后的长表与可比性结论
  POST /advice              上传 → 只做算力预估，不建模

设计上坚持两件事：
  1) 不静默改结论：任何降预算、子采样、不可比字段都会在返回里写明；
  2) 出错给原因：数据问题（没有标签列、映射对不上）返回 4xx 并说明哪一步失败，
     而不是 500。
"""

from __future__ import annotations

import io
import time

from hybridscope import __version__

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from pydantic import BaseModel, Field
except Exception as _exc:                                  # pragma: no cover
    raise ImportError(
        "服务层需要 fastapi 与 pydantic：pip install fastapi uvicorn pydantic"
    ) from _exc

import numpy as np
import pandas as pd

from hybridscope.estimator import HybridScopeEstimator
from hybridscope.io import split_columns
from hybridscope.prep import Prep

app = FastAPI(title="HybridScope",
              description="混合属性分型与权重可估计性诊断",
              version=__version__)


# ---------------------------------------------------------------- 数据结构

class AnalyzeOptions(BaseModel):
    target: str | None = None
    cat: list[str] = Field(default_factory=list)
    k: int = 5
    seed: int = 0
    fdr_alpha: float = 0.05
    boot: int = 200
    stab_boot: int = 60
    stab_perm: int = 60
    no_stability: bool = False
    speed: str = "auto"
    max_n: int | None = None
    prep: bool = True
    field_schema: dict | None = None
    profile: str | None = None
    name: str | None = None
    band: int | None = None


class TextRequest(AnalyzeOptions):
    csv: str
    sep: str = ","


class CompareRequest(BaseModel):
    a: dict
    b: dict
    alpha: float = 0.05
    field_schema: dict | None = None


def _clean(o):
    """NaN / Inf 转成 null：JSON 不认它们，浏览器与多数客户端也解析不了。

    诊断里标准误为 NaN（权重退化）、q 为 NaN（不可识别）都是正常状态，
    直接丢给 json.dumps 会让整个响应变成 500。
    """
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _read(upload: UploadFile, data: bytes) -> pd.DataFrame:
    name = (upload.filename or "").lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig",
                       on_bad_lines="skip")


def _apply_schema(df: pd.DataFrame, sch: dict, profile: str | None,
                  target: str | None):
    """按字段映射折叠，返回 (df, schema, profile, audit)。"""
    from hybridscope.schema import Schema
    schema = Schema.from_dict(sch)
    prof = profile or (list(schema.profiles)[0] if len(schema.profiles) == 1 else None)
    if prof not in schema.profiles:
        cand = [k for k in schema.profiles if prof and k in str(prof)]
        prof = cand[0] if cand else None
    if prof not in schema.profiles:
        raise HTTPException(
            status_code=422,
            detail="映射里没有批次 %r，已有：%s" % (profile, list(schema.profiles)))
    uni = schema.apply(df, prof)
    if not uni.ok():
        raise HTTPException(status_code=422,
                            detail="字段映射未通过：%s" % "；".join(uni.report["errors"]))
    out = uni.df
    feat = [f.name for f in schema.fields if f.role == "feature" and f.name in out.columns]
    cat_cols = [f.name for f in schema.fields
                if f.role == "feature" and f.kind == "categorical" and f.name in out.columns]
    if target is None:
        tg = [f.name for f in schema.fields if f.role == "target" and f.name in out.columns]
        target = tg[0] if tg else None
    if target and target in out.columns:
        import pandas.api.types as ptypes
        v = pd.to_numeric(out[target], errors="coerce")
        if v.notna().all() and v.nunique() > 8:
            # 连续的总分不能直接当标签：按三分位切档，否则 kNN 会把每个分数
            # 当成一个类，准确率必然很低。
            out["__band__"] = pd.qcut(v, 3, labels=["低", "中", "高"],
                                      duplicates="drop").astype(str)
            target = "__band__"
    keep = [c for c in feat + ([target] if target else []) if c in out.columns]
    return out[keep], schema, prof, uni.report, cat_cols, target


def _analyze(df: pd.DataFrame, opt: AnalyzeOptions):
    """核心分析：体检 → 拟合 → 诊断 → 快照，返回可序列化的 dict。"""
    from hybridscope.monitor import snapshot as snap_of
    from hybridscope.speed import advice as advice_text

    t0 = time.time()
    schema = None
    profile = opt.profile
    audit = None
    cat_cols = list(opt.cat)
    target = opt.target
    if opt.field_schema:
        df, schema, profile, audit, cat_cols, target = _apply_schema(
            df, opt.field_schema, opt.profile, opt.target)

    roles = {c: ("cat" if c in cat_cols else "num") for c in df.columns}
    warnings: list[str] = []
    # 标签是连续分数（总评 0–100 有几十个取值）时，kNN 会把每个分数当成一类，
    # 准确率必然很低。要么按分位数切档，要么至少把这件事说出来。
    if target and target in df.columns:
        v = pd.to_numeric(df[target], errors="coerce")
        if v.notna().all() and int(v.nunique()) > 8:
            nb = int(opt.band) if opt.band else 3
            labels = ["低", "中", "高", "很高", "极高"][:nb]
            if opt.band:
                src = target
                df["__band__"] = pd.qcut(v, nb, labels=labels,
                                         duplicates="drop").astype(str)
                target = "__band__"
                # 原始分数必须剔掉：档位由它切出来，留着就是标签泄漏，
                # 权重会假性显著（实测「总评」能拿到 0.41 的权重）。
                df = df.drop(columns=[src])
                warnings.append("标签列 %s 是连续分数（%d 个取值），已按分位数切成 %d 档，"
                                "原列已从特征中剔除（否则标签泄漏）"
                                % (src, int(v.nunique()), nb))
            else:
                warnings.append(
                    "标签列有 %d 个不同取值，看着是连续分数而不是分型标签：kNN 会把每个"
                    "分数当成一类，准确率会很低。要自动切档请给 band=3。" % int(v.nunique()))
        elif not pd.api.types.is_numeric_dtype(df[target]):
            from sklearn.preprocessing import LabelEncoder
            df[target] = LabelEncoder().fit_transform(df[target].astype(str))
    df_raw = df.copy()

    prep = None
    t_prep = None
    if opt.prep:
        t1 = time.time()
        prep = Prep()
        df = prep.fit_transform(df, target=target, cat_cols=cat_cols or None)
        t_prep = time.time() - t1

    X_num, X_cat, num_names, cat_names, y = split_columns(
        df, target=target, cat_cols=cat_cols or None)
    if y is None:
        raise HTTPException(status_code=422,
                            detail="没有标签列：请传 target，或在字段映射里把某个字段的 role 设为 target")

    keep = []
    for key, val in (("pso_particles", None), ("pso_iters", None),
                     ("n_boot", opt.boot), ("stab_boot", opt.stab_boot),
                     ("stab_perm", opt.stab_perm)):
        keep.append(key)
    est = HybridScopeEstimator(
        k=opt.k, seed=opt.seed, fdr_alpha=opt.fdr_alpha, n_boot=opt.boot,
        run_stability=not opt.no_stability, stab_boot=opt.stab_boot,
        stab_perm=opt.stab_perm, speed=opt.speed, max_n=opt.max_n,
        speed_keep=("n_boot", "stab_boot", "stab_perm"), auto_prep=False)
    est.fit_hybrid(X_num, X_cat, y, feature_names=list(num_names) + list(cat_names))

    s = est.summary()
    out = {
        "version": __version__,
        "seconds": round(time.time() - t0, 3),
        "n": s.get("n"),
        "k": s.get("k"),
        "target": target,
        "profile": profile,
        "metrics": s.get("metrics"),
        "weights": s.get("weights"),
        "advisor": s.get("advisor"),
        "diagnostics": s.get("diagnostics"),
        "fdr": s.get("fdr"),
        "identifiable": s.get("identifiable"),
        "stability": s.get("stability"),
        "speed": s.get("speed"),
        "prep_seconds": (round(t_prep, 3) if t_prep is not None else None),
        "schema_audit": audit,
        "warnings": warnings,
    }
    if prep is not None:
        out["prep"] = prep.report()
    out["snapshot"] = snap_of(est, df=df_raw, name=opt.name or profile,
                              source=opt.name or profile, schema=schema,
                              profile=profile)
    if getattr(est, "speed_", None) and est.speed_.get("mode") != "thorough":
        warnings.append(
            "本次用的是 %s 预算（PSO %d×%d）。权重是随机搜索的解，预算小则解更粗："
            "换种子重跑的离散度会偏大，跨批次比较时建议两边都用 thorough，"
            "或固定预算后重跑几次取中位数。"
            % (est.speed_.get("mode"), est.speed_.get("pso_particles"),
               est.speed_.get("pso_iters")))
    out["advice"] = advice_text(
        s.get("n", 0), max(1, len(num_names)), len(cat_names),
        run_stability=not opt.no_stability)
    return _clean(out)


# ---------------------------------------------------------------- 路由

@app.get("/health")
def health():
    return {"ok": True, "version": __version__,
            "service": "hybridscope", "docs": "/docs"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...),
                  target: str | None = Form(None),
                  cat: str | None = Form(None),
                  k: int = Form(5),
                  seed: int = Form(0),
                  fdr_alpha: float = Form(0.05),
                  boot: int = Form(200),
                  stab_boot: int = Form(60),
                  stab_perm: int = Form(60),
                  no_stability: bool = Form(False),
                  speed: str = Form("auto"),
                  max_n: int | None = Form(None),
                  prep: bool = Form(True),
                  profile: str | None = Form(None),
                  name: str | None = Form(None),
                  schema_file: UploadFile | None = File(None)):
    """上传一张表，返回完整分析。schema_file 为字段映射 JSON（可选）。"""
    data = await file.read()
    try:
        df = _read(file, data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="读表失败：%s" % exc)
    sch = None
    if schema_file is not None:
        import json
        try:
            sch = json.loads((await schema_file.read()).decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=422, detail="字段映射 JSON 解析失败：%s" % exc)
    opt = AnalyzeOptions(target=target, cat=[c for c in (cat or "").split(",") if c],
                         k=k, seed=seed, fdr_alpha=fdr_alpha, boot=boot,
                         stab_boot=stab_boot, stab_perm=stab_perm,
                         no_stability=no_stability, speed=speed, max_n=max_n,
                         prep=prep, field_schema=sch, profile=profile, name=name)
    try:
        return _analyze(df, opt)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="分析失败：%s" % exc)


@app.post("/analyze/text")
def analyze_text(req: TextRequest):
    """CSV 文本体直接分析，便于程序调用（不用 multipart）。"""
    try:
        df = pd.read_csv(io.StringIO(req.csv), sep=req.sep)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="CSV 解析失败：%s" % exc)
    try:
        return _analyze(df, req)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="分析失败：%s" % exc)


@app.post("/snapshot")
async def snapshot_ep(file: UploadFile = File(...),
                      target: str | None = Form(None),
                      cat: str | None = Form(None),
                      name: str | None = Form(None),
                      k: int = Form(5),
                      speed: str = Form("fast"),
                      no_stability: bool = Form(True),
                      schema_file: UploadFile | None = File(None)):
    """只返回快照：存档用，之后拿两份快照调 /compare。"""
    res = await analyze(file=file, target=target, cat=cat, k=k, speed=speed,
                        no_stability=no_stability, name=name,
                        schema_file=schema_file)
    return _clean(res["snapshot"])


@app.post("/compare")
def compare(req: CompareRequest):
    """两份快照 → 漂移结论（真漂移 / 只是噪声 / 数据先变了 / 口径不可比）。"""
    from hybridscope.monitor import compare as cmp, to_text
    from hybridscope.schema import Schema
    sch = Schema.from_dict(req.field_schema) if req.field_schema else None
    try:
        rep = cmp(req.a, req.b, alpha=req.alpha, schema=sch)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="对比失败：%s" % exc)
    return _clean({"kind": rep["kind"], "headline": rep["headline"],
            "rows": rep["rows"], "data": rep["data"],
            "blocked": rep.get("blocked"), "caveats": rep.get("caveats"),
            "budget_note": rep.get("budget_note"),
            "schema_note": rep.get("schema_note"),
            "chi2": rep.get("chi2"), "p_global": rep.get("p_global"),
            "text": to_text(rep)})


@app.post("/unify")
async def unify(files: list[UploadFile] = File(...),
                schema_file: UploadFile = File(...),
                profiles: str | None = Form(None)):
    """多张表按同一映射对齐：返回长表 CSV 与可比性结论。"""
    import json
    from hybridscope.schema import Schema, audit_text, compat_text, harmonize
    try:
        sch = Schema.from_dict(json.loads((await schema_file.read()).decode("utf-8")))
    except Exception as exc:
        raise HTTPException(status_code=422, detail="字段映射 JSON 解析失败：%s" % exc)
    names = [p.strip() for p in (profiles or "").split(",") if p.strip()]
    frames, audits = {}, []
    for i, f in enumerate(files):
        prof = names[i] if i < len(names) else (f.filename or "").rsplit(".", 1)[0]
        if prof not in sch.profiles:
            cand = [k for k in sch.profiles if k in str(prof)]
            prof = cand[0] if cand else prof
        if prof not in sch.profiles:
            raise HTTPException(status_code=422,
                                detail="%s 在映射里找不到批次，已有：%s"
                                       % (prof, list(sch.profiles)))
        frames[prof] = _read(f, await f.read())
    for prof, fr in frames.items():
        audits.append(audit_text(sch.apply(fr, prof)))
    cp = sch.compatibility(frames)
    long, _ = harmonize(sch, frames)
    return _clean({"profiles": sorted(frames), "rows": int(len(long)),
            "columns": list(long.columns),
            "csv": long.to_csv(index=False),
            "audits": audits, "compatibility": cp,
            "text": compat_text(cp)})


@app.post("/advice")
async def advice(file: UploadFile = File(...),
                 target: str | None = Form(None),
                 cat: str | None = Form(None),
                 boot: int = Form(200),
                 stab_boot: int = Form(60),
                 stab_perm: int = Form(60)):
    """只做算力预估，不建模。"""
    from hybridscope.speed import advice as advice_text, budget
    df = _read(file, await file.read())
    cats = [c for c in (cat or "").split(",") if c]
    m = max(1, sum(1 for c in df.columns
                   if c != target and c not in cats
                   and pd.api.types.is_numeric_dtype(df[c])))
    n = len(df)
    return _clean({"n": n, "m": m,
            "text": advice_text(n, m, len(cats), stab_boot=stab_boot,
                                stab_perm=stab_perm, n_boot=boot),
            "budgets": {mode: budget(n, m, len(cats), mode)
                        for mode in ("auto", "fast", "thorough")}})


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="hybridscope-serve")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args(argv)
    import uvicorn
    uvicorn.run("hybridscope.serve:app", host=args.host, port=args.port,
                reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
