# -*- coding: utf-8 -*-
"""设备运维模板：传感器统计特征 → 设备状态分型。

运维表有几个和成绩表完全不同的坑，模板的价值就是把它们提前说出来：
  1) 同一台设备被采样很多次 → 这些行不是独立样本。sandwich 方差假设样本独立，
     重复测量会让它低估标准误，于是「显著」是假的。模板数出重复单元并提示。
  2) 故障样本通常极少（正常 95% / 故障 5%）。这种不平衡下准确率是虚高指标，
     一个全判正常的模型也有 95% 准确率，权重结论同样不可信。
  3) 传感器的量纲差着好几个数量级（温度几十、振动几万），不处理的话
     大数值通道会独占距离度量——数据体检会自动标准化，但要在报告里写明。
  4) 时间戳不是特征。把它当数值喂进去，等于让模型按「采集先后」分型。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

ID_HINTS = ("设备", "机台", "编号", "资产", "序号", "时间戳", "采集时间",
            "时间", "日期", "asset", "device", "id", "ID", "timestamp")
CAT_HINTS = ("型号", "产线", "工况", "车间", "厂商", "等级", "批次", "站点",
             "线别", "类别", "type", "line")
TARGET_HINTS = ("状态", "故障", "是否故障", "告警", "标签", "结论",
                "status", "fault", "label", "y")
TIME_HINTS = ("时间", "日期", "时间戳", "time", "date", "timestamp")

# 传感器通道的语义归类，用于把权重翻译成人话
TRAITS = [
    (("温度", "temp", "℃"), "热状态"),
    (("振动", "加速度", "vib", "acc"), "机械振动"),
    (("电流", "功率", "电压", "curr", "power", "volt"), "电气负荷"),
    (("压力", "压强", "press"), "压力/流体"),
    (("转速", "频率", "speed", "rpm", "freq"), "转速/频率"),
    (("噪声", "声", "noise", "db"), "声学"),
    (("位移", "间隙", "间隙量"), "形变/间隙"),
    (("湿度", "humid"), "环境湿度"),
]


def detect_roles(df: pd.DataFrame, id_col: str | None = None) -> dict:
    """角色识别：设备编号与时间戳是标识，状态是标签，型号/产线是分组。

    与成绩模板不同，这里先判 target 再判 cat：设备的「状态」列同时命中
    两类关键词，判反了就没有标签可用。
    """
    roles = {}
    for c in df.columns:
        name = str(c)
        low = name.lower()
        if id_col and c == id_col:
            roles[c] = "id"
        elif any(h in name for h in TARGET_HINTS):
            roles[c] = "target"
        elif any(h in name for h in ID_HINTS) or any(h in low for h in TIME_HINTS):
            roles[c] = "id"
        elif any(h in name for h in CAT_HINTS) or not pd.api.types.is_numeric_dtype(df[c]):
            roles[c] = "cat"
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


def class_balance(df: pd.DataFrame, target: str, minority_flag: float = 0.10
                  ) -> dict:
    """类别占比。少数类占比过低时，准确率与权重都不可信。"""
    vc = df[target].astype(str).value_counts()
    share = (vc / max(len(df), 1)).to_dict()
    smallest = float(vc.min()) / max(len(df), 1)
    return {
        "counts": {str(k): int(v) for k, v in vc.items()},
        "share": {str(k): float(v) for k, v in share.items()},
        "n_classes": int(vc.size),
        "minority_share": smallest,
        "imbalanced": bool(vc.size > 1 and smallest < minority_flag),
    }


def repeated_units(df: pd.DataFrame, id_col: str | None = None) -> dict:
    """找出重复测量：同一台设备出现多次 → 样本不独立，标准误会低估。"""
    roles = detect_roles(df, id_col=id_col)
    ids = [c for c, r in roles.items() if r == "id" and c != id_col] or None
    if id_col is None and ids:
        id_col = ids[0]
    if not id_col or id_col not in df.columns:
        return {"id_column": None, "n_units": None, "max_repeat": None,
                "repeat_ratio": None, "independent": True}
    vc = df[id_col].astype(str).value_counts()
    n_rows, n_units = int(len(df)), int(vc.size)
    max_rep = int(vc.max())
    return {
        "id_column": id_col,
        "n_units": n_units,
        "n_rows": n_rows,
        "max_repeat": max_rep,
        "repeat_ratio": float(n_rows / max(n_units, 1)),
        # 一行一个单元才算独立；有重复就必须承认标准误偏乐观
        "independent": bool(max_rep <= 1),
    }


def summarize(df: pd.DataFrame, target: str | None = None,
              roles: dict | None = None, id_col: str | None = None) -> dict:
    roles = roles or detect_roles(df, id_col=id_col)
    target = target or pick_target(df, roles)
    num, cat = feature_split(df, target=target, roles=roles)
    out = {
        "n_rows": int(len(df)),
        "target": target,
        "numeric_channels": num,
        "categorical": cat,
        "id_columns": [c for c, r in roles.items() if r == "id"],
        "scale_spread": None,
        "traits": {c: trait_of(c) for c in num},
    }
    if num:
        sd = df[num].apply(pd.to_numeric, errors="coerce").std(ddof=1)
        sd = sd[sd > 0]
        if len(sd) >= 2:
            out["scale_spread"] = float(sd.max() / max(sd.min(), 1e-12))
    if target:
        out["balance"] = class_balance(df, target)
    out["repeated"] = repeated_units(df, id_col=id_col)
    return out


def trait_of(col: str) -> str:
    name = str(col).lower()
    best, blen = None, 0
    for keys, trait in TRAITS:
        for k in keys:
            if k.lower() in name and len(k) > blen:
                best, blen = trait, len(k)
    return best or "未归类通道"


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
            tail = {"green": "不同 %s 的设备状态确实分得开" % name,
                    "amber": "该分组方向可信但精度不足",
                    "red": "看不出 %s 对状态分型有作用" % name}.get(
                        lvl, "权重在边界上，等价于该分组未参与分型")
        else:
            if lvl == "green":
                tail = "该通道对状态分型的贡献可被数据支持"
            elif lvl == "amber":
                tail = "方向可信但精度不足"
            elif lvl == "red":
                tail = ("多重检验校正后不显著（q=%s），当前样本量下无法把它和"
                        "「各通道等权」区分开"
                        % ("%.3f" % q if isinstance(q, float) else "n/a"))
            else:
                tail = "权重落在边界上，等价于该通道未参与分型"
        share = ""
        if rows:
            tot = sum(abs(float(x.get("weight", 0.0))) for x in rows) or 1.0
            share = "占权重 %.0f%%，" % (abs(w) / tot * 100)
        out.append({"feature": name, "trait": trait,
                    "text": "%s（%s）权重 %.3f，%sz=%.2f：%s。"
                            % (name, trait, w, share, z, tail)})
    return out


def caveats(summ: dict) -> list[str]:
    """把这份运维表该注意的事列出来，直接进报告的动作清单。"""
    out = []
    rep = summ.get("repeated") or {}
    if rep.get("id_column") and not rep.get("independent", True):
        out.append("同一设备被采样多次（%d 行 / %d 台，最多 %d 次）：行之间不独立，"
                   "sandwich 标准误会偏乐观，权重「显著」要打折看。"
                   % (rep["n_rows"], rep["n_units"], rep["max_repeat"]))
    bal = summ.get("balance") or {}
    if bal.get("imbalanced"):
        small = min(bal["share"], key=bal["share"].get)
        out.append("类别极不平衡（最小的「%s」只占 %.1f%%）：准确率是虚高指标，"
                   "权重结论也不可靠，先看清楚少数类到底有多少样本。"
                   % (small, 100 * bal["minority_share"]))
    if summ.get("scale_spread") and summ["scale_spread"] > 10:
        out.append("通道量纲相差 %.0f 倍：必须标准化，否则大数值通道独占距离度量。"
                   % summ["scale_spread"])
    return out


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path) if str(path).lower().endswith(".csv") else pd.read_excel(path)
    df.columns = [re.sub(r"^﻿", "", str(c)).strip() for c in df.columns]
    return df
