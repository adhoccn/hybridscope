# -*- coding: utf-8 -*-
"""HybridScope 工作台：streamlit run hybridscope/app.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import streamlit as st

from hybridscope.advisor import acf, profile_ratio
from hybridscope.io import concat_frames, merge_frames, split_columns, table_overview
from hybridscope.estimator import HybridScopeEstimator
from hybridscope.prep import Prep
from hybridscope.schema import Schema
from hybridscope.stability import to_text as stability_text
from hybridscope import templates
from knn_hybrid.data import add_noise, make_hybrid_dataset

COLOR = {"green": "#2e7d32", "amber": "#f9a825", "red": "#c62828", "boundary": "#9e9e9e"}
TAU_ON, TAU_OFF = 0.8, 0.5   # 论文 A 的 τ̂=acf(8)/acf(1) 判据阈值
MODE_LABEL = {"concat": "纵向合并", "merge": "横向合并", "each": "逐个分析"}

st.set_page_config(page_title="HybridScope", layout="wide")
st.title("HybridScope")
st.caption("混合属性分型引擎 · 权重可估计性诊断")

LETTERS = np.array(list("ABCDEFGH"))


def pca_frame(X):
    """数值矩阵的前两主成分，用于目视分型结构。"""
    Xf = np.asarray(X, float)
    if Xf.shape[1] < 2 or Xf.shape[0] < 3:
        return None
    Z = (Xf - Xf.mean(0)) / Xf.std(0).clip(1e-12)
    P = PCA(n_components=2).fit_transform(Z)
    return np.clip(P, np.percentile(P, 1), np.percentile(P, 99))


def read_any(up) -> pd.DataFrame:
    return pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)


def demo_frame(n=240, snr=4.0, seed=0):
    d = make_hybrid_dataset(n_samples=n, d_num=4, d_cat=3, n_clusters=3, seed=seed)
    d = add_noise(d, snr_db=snr, seed=seed + 1)
    df = pd.DataFrame(d.X_num, columns=[f"传感器{i + 1}" for i in range(d.X_num.shape[1])])
    for j in range(d.X_cat.shape[1]):
        df[f"工况{j + 1}"] = LETTERS[d.X_cat[:, j]]
    df["分型标签"] = d.y
    return df


def demo_grades(n=180, seed=0):
    """模拟学情数据：4 次测验 + 4 次实验报告 + 专业，标签为学情档位。"""
    rng = np.random.default_rng(seed)
    t = rng.normal(size=n)
    load_t = np.array([0.55, 0.60, 0.65, 0.90])   # 测验 4 区分度最强
    load_r = np.array([0.45, 0.40, 0.12, 0.50])   # 报告 3 区分度不足
    tests = np.clip(72 + 13 * np.outer(t, load_t) + rng.normal(0, 6, (n, 4)), 0, 100)
    reps = np.clip(78 + 9 * np.outer(t, load_r) + rng.normal(0, 5, (n, 4)), 0, 100)
    df = pd.DataFrame(np.round(tests), columns=[f"测验{i + 1}" for i in range(4)])
    for j in range(4):
        df[f"报告{j + 1}"] = np.round(reps[:, j])
    df["专业"] = rng.choice(["计算机", "自动化", "电子信息", "机械"], n)
    total = 0.5 * tests.mean(1) + 0.5 * reps.mean(1)
    df["学情档位"] = pd.cut(np.clip(total, 0, 100), [-1, 70, 85, 101],
                           labels=["需关注", "合格", "优秀"]).astype(str)
    return df


def render_overview(df: pd.DataFrame):
    """样本量/字段/缺失/类别数四张卡 + 数值列箱线图。"""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("样本数", len(df))
    c2.metric("字段数", df.shape[1])
    c3.metric("缺失值", int(df.isna().sum().sum()))
    c4.metric("类别字段", sum(1 for c in df.columns
                              if not pd.api.types.is_numeric_dtype(df[c])))

    id_like = [c for c in df.columns
               if len(df) > 3 and df[c].nunique(dropna=False) >= 0.95 * len(df)]
    if id_like:
        st.warning("这些列几乎每行一个取值（多半是学号/编号/时间戳）："
                   + "、".join(id_like)
                   + "。当类别特征用时每个样本自成一类，会把距离尺度冲掉，"
                     "建议不要勾选为特征。")

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    sub = df[num_cols].apply(pd.to_numeric, errors="coerce") if num_cols else None
    if sub is not None and len(sub.columns):
        z = (sub - sub.mean()) / sub.std().replace(0, 1)
        fig = go.Figure()
        for c in sub.columns:
            fig.add_trace(go.Box(y=z[c], name=c, boxpoints="outliers",
                                 marker_size=3, showlegend=False))
        fig.update_layout(height=300, yaxis_title="标准化取值",
                          title="数值字段分布（箱线图）",
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("箱线图用于发现量纲差异与异常值。分型前引擎会按列标准化，"
                   "因此量纲不影响结果，但极端离群点会拉动距离尺度。")

    with st.expander("查看原始数据前 20 行"):
        st.dataframe(df.head(20), width="stretch", hide_index=True)


def render_merge_report(rep: dict):
    """多表合并后发生了什么：丢了哪些列、补了多少空、对上了多少行。"""
    with st.expander("合并明细（建议先看一眼）", expanded=True):
        if rep["mode"] == "concat":
            st.write(f"合并 {len(rep['files'])} 个文件，共 **{rep['n_total']}** 行；"
                     f"对齐方式：**{'只保留共有列' if rep['keep'] == 'intersection' else '保留全部列'}**。")
            rows = [{"文件": n, "原行数": rep["n_rows"][n]} for n in rep["files"]]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            if rep["dropped_cols"]:
                st.warning("这些列不是所有文件都有，已排除："
                           + "、".join(rep["dropped_cols"]))
            if rep["filled"]:
                txt = "；".join(f"{n} 缺 {len(v)} 列（已填缺失）" for n, v in rep["filled"].items())
                st.warning("保留全部列模式下，缺失部分填为空值，分型前这些行会被整行丢弃：\n\n" + txt)
            if rep["conflicts"]:
                st.warning("同名列在不同文件里类型不一致（数值 vs 文本），已统一按类别处理："
                           + "、".join(rep["conflicts"]))
            if rep["source_col"]:
                st.caption(f"已加入「{rep['source_col']}」列标记每行来自哪个文件。"
                           "若各文件是不同批次（不同学期/不同设备），建议把它勾选为类别列——"
                           "否则批次差异会被算进特征权重里。")
        else:
            st.write(f"按主键「{rep['key']}」做 **{rep['how']}** 连接，"
                     f"结果 **{rep['n_total']}** 行 × {rep['n_cols']} 列。")
            rows = []
            for n, v in rep["key_stats"].items():
                rows.append({"文件": n, "行数": v["rows"], "主键唯一值": v["unique"],
                             "主键重复": v["dup"], "主键缺失": v["nan"]})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            for stp in rep["steps"]:
                st.write(f"接入 {stp['file']}：{stp['before']} → {stp['after']} 行")
            dup = sum(v["dup"] for v in rep["key_stats"].values())
            if dup:
                st.warning(f"主键在部分文件里有重复值（合计 {dup} 处），"
                           "连接后行数会膨胀，请先去重或换一个真正唯一的字段。")
            st.caption("同名但非主键的列会自动加「@文件名」后缀，不会互相覆盖。")


def analyze(df, y_col, cat_cols, k, denoise_mode, particles, iters,
            auto_prep=True, fdr_alpha=0.05, run_stability=True, speed="off"):
    """跑一次完整拟合，返回结果包。

    auto_prep=True 时先做数据体检与修补（缺失填补 / 离群截断 / 低频档合并 /
    标识列剔除 / 量纲失衡时标准化），不再整行丢弃有缺失的样本。
    speed != "off" 时由 hybridscope.speed 按样本量定 PSO 规模与置换次数，
    上面手调的粒子/迭代会被预算覆盖（除非显式档位是 off）。
    """
    est = HybridScopeEstimator(
        k=k, denoise={"auto": "auto", "on": True, "off": False}[denoise_mode],
        pso_particles=particles, pso_iters=iters, seed=0,
        auto_prep=False, fdr_alpha=fdr_alpha, run_stability=run_stability,
        speed=speed)
    prep = None
    if auto_prep:
        prep = Prep()
        df = prep.fit_transform(df, target=y_col, cat_cols=cat_cols)
    X_num, X_cat, num_names, cat_names, yy = split_columns(df, target=y_col,
                                                           cat_cols=cat_cols)
    est.fit_hybrid(X_num, X_cat, yy, feature_names=num_names + cat_names)
    # 标签编码 0..C-1 与原始取值的对照；split_columns 内部先 dropna，这里保持一致
    used = df.dropna().reset_index(drop=True)
    levels = None
    if y_col is not None and y_col in df.columns:
        levels = df[y_col].astype("category").cat.categories.tolist()
    return {"est": est, "s": est.summary(), "X_num": X_num, "num_names": num_names,
            "cat_names": cat_names, "levels": levels, "df_used": used,
            "prep": None if prep is None else prep.report()}


def _tmpl_caveats(tmpl, summ: dict) -> list[str]:
    """模板的注意事项。各模板 caveats 签名不一（survey 多一个审计参数），容错调用。"""
    fn = getattr(tmpl, "caveats", None)
    if fn is None:
        return []
    for args in ((summ, None), (summ,)):
        try:
            return [str(c) for c in (fn(*args) or [])]
        except TypeError:
            continue
        except Exception:
            return []
    return []


def render_prep(rep):
    """数据体检结论：剔了什么、补了什么、为什么。"""
    if not rep:
        return
    with st.expander("数据体检（点击展开）", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("原始行数", rep["n_rows_in"])
        c2.metric("剔除列", rep["n_dropped_cols"])
        c3.metric("重复行", rep["n_dup_rows_dropped"])
        if rep.get("dropna_would_drop"):
            st.warning("沿用「有缺失就整行丢弃」会损失 %d 行，体检后改为逐列修补。"
                       % rep["dropna_would_drop"])
        if rep.get("dropped"):
            st.markdown("**剔除的列**")
            st.dataframe(pd.DataFrame(rep["dropped"]), width="stretch",
                         hide_index=True)
        if rep.get("actions"):
            st.markdown("**修补动作**")
            st.dataframe(pd.DataFrame(rep["actions"]), width="stretch",
                         hide_index=True)
        if rep.get("scaled"):
            st.caption("数值列量级相差过大，已统一标准化后再进距离。")


def label_name(levels, c):
    """簇/类编号 → 原始取值（如 0 → 需关注）。"""
    if levels is not None and 0 <= c < len(levels):
        return str(levels[c])
    return str(c)


def cluster_profile(est, res):
    """无监督场景：算出每个簇在数值特征上的 z 偏移与类别列的众数。"""
    Xn = np.asarray(est.X_num_, float)
    labels = np.asarray(est.labels_, int)
    num_names = list(res["num_names"])[: Xn.shape[1]]
    mu = Xn.mean(0)
    sd = Xn.std(0)
    sd[sd < 1e-12] = 1.0
    Z = (Xn - mu) / sd
    clusters = sorted(set(labels.tolist()))
    zm = np.stack([Z[labels == c].mean(0) for c in clusters]) if clusters else np.zeros((0, 0))

    wnum = np.asarray(est.w_, float) if est.w_ is not None else None
    if wnum is not None and len(wnum) >= len(num_names):
        wnum = wnum[: len(num_names)]
        score = zm @ wnum / max(wnum.sum(), 1e-12)
    else:
        score = zm.mean(1) if zm.size else np.zeros(len(clusters))

    used = res["df_used"]
    rows = []
    for i, c in enumerate(clusters):
        m = labels == c
        order = np.argsort(-np.abs(zm[i]))[:3]
        traits = "、".join(f"{num_names[j]} {zm[i][j]:+.2f}σ" for j in order)
        modes = []
        for cn in res["cat_names"]:
            if cn in used.columns and m.sum():
                vc = used.loc[m, cn].value_counts(normalize=True)
                if len(vc):
                    modes.append(f"{cn} 多为 {vc.index[0]}（{vc.iloc[0] * 100:.0f}%）")
        rows.append({
            "编号": int(c),
            "样本数": int(m.sum()),
            "占比": f"{m.sum() / len(labels) * 100:.1f}%",
            "综合位置": round(float(score[i]), 3),
            "最突出的三项": traits,
            "类别特征": "；".join(modes) if modes else "—",
        })
    return clusters, zm, score, pd.DataFrame(rows), num_names


def render_cluster_profile(est, res):
    """把簇编号翻译成「这一簇是什么样的样本」."""
    clusters, zm, score, tbl, num_names = cluster_profile(est, res)
    st.subheader("簇画像：每个编号对应什么样的样本")

    n_obs = int(tbl["样本数"].sum())
    min_size = max(5, int(np.ceil(0.02 * n_obs)))
    size = tbl["样本数"].to_numpy(int)
    main_idx = [i for i in range(len(clusters)) if size[i] >= min_size]
    frag_idx = [i for i in range(len(clusters)) if size[i] < min_size]

    names = {1: ["全体"], 2: ["偏低型", "偏高型"],
             3: ["偏低型", "中间型", "偏高型"],
             4: ["最低型", "偏低型", "偏高型", "最高型"],
             5: ["最低型", "偏低型", "中间型", "偏高型", "最高型"]}.get(len(main_idx))
    alias = {}
    if names:
        for i, j in enumerate(np.argsort(score[main_idx])):
            alias[int(clusters[main_idx[j]])] = names[i]
    tbl.insert(1, "建议称呼",
               [alias.get(int(c), "—") for c in tbl["编号"]])

    st.dataframe(tbl.iloc[main_idx] if main_idx else tbl,
                 width="stretch", hide_index=True)
    if not names and len(main_idx) > 5:
        st.caption("主簇超过 5 个时编号命名没有意义，请直接看「综合位置」"
                   "（按学到的权重加权，越大越偏上）与「最突出的三项」。")
    if len(main_idx) <= 1:
        st.warning("样本几乎连成一片，这批特征在当前 k 下没有可分结构；"
                   "可以调小 k、换特征，或改用有标签路径。")

    if frag_idx:
        st.warning(
            f"另有 {len(frag_idx)} 个碎片簇、共 {int(size[frag_idx].sum())} 条样本"
            f"（每簇不足 {min_size} 条）：它们是 k 近邻图上的离群点，不是真正的分型。"
            "建议调大 k 近邻让它们并入主簇，或先把离群样本剔掉再跑。")
    if main_idx and size[main_idx].max() >= 0.6 * n_obs:
        st.warning(f"最大一簇占了 {size[main_idx].max() / n_obs * 100:.1f}% 的样本："
                   "k 近邻图基本连通，当前 k 与特征组合还没切出有意义的分型。"
                   "可以调小 k、换一批特征，或改走有标签路径。")

    if zm.size and len(num_names) and main_idx:
        cols = [i for i in main_idx]
        f = go.Figure(go.Heatmap(
            z=zm[cols].T,
            x=[f"簇{clusters[i]}"
               + (f" · {alias[int(clusters[i])]}" if int(clusters[i]) in alias else "")
               for i in cols],
            y=num_names, colorscale="RdBu", reversescale=True, zmid=0,
            colorbar=dict(title="偏离总体均值（σ）"),
            hovertemplate="%{y}<br>%{x}：%{z:+.2f}σ<extra></extra>"))
        f.update_layout(height=max(300, 60 * len(num_names) + 120),
                        title="各簇特征偏移（红=高于总体均值，蓝=低于）",
                        margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(f, width="stretch")

    s_main = score[main_idx] if main_idx else score
    lo, hi = float(s_main.min()), float(s_main.max())
    lines = []
    for i in main_idx if main_idx else range(len(clusters)):
        c = clusters[i]
        rel = (float(score[i]) - lo) / (hi - lo) if hi > lo else 0.5
        tag = ("主簇中位置最高" if rel > 0.66
               else "主簇中位置最低" if rel < 0.33 else "主簇中位置居中")
        head = f"簇 {c}" + (f"（{alias[int(c)]}）" if int(c) in alias else "")
        lines.append(f"- **{head}，{int(tbl['样本数'][i])} 条，"
                     f"占 {tbl['占比'][i]}**：{tbl['最突出的三项'][i]}，{tag}。")
    st.markdown("**逐簇解读**\n" + "\n".join(lines))
    st.caption("σ 是该簇均值相对全体均值偏离多少个标准差；「综合位置」按学到的特征权重加权，"
               "仅用于给簇排序和命名，不代表优劣。簇编号本身没有含义，"
               "换一批数据同样的编号会指向不同的簇——请看画像而不是编号。")


def render_report(res, denoise_mode: str):
    """零、数据体检 → 一、去噪顾问 → 二、分型质量 → 三、权重可信度 → 四、解读。"""
    render_prep(res.get("prep"))
    est, s, X_num, num_names = res["est"], res["s"], res["X_num"], res["num_names"]
    levels = res["levels"]
    a = est.advisor_
    tau = profile_ratio(np.asarray(X_num, float))
    acf1 = np.array([acf(np.asarray(X_num, float)[:, j], 1)
                     for j in range(np.asarray(X_num, float).shape[1])])

    st.subheader("一、去噪顾问：这批数据该不该做沿样本维小波去噪")
    if a is not None:
        msg = "建议%s沿样本维小波去噪" % ("启用" if a["denoise"] else "关闭")
        (st.success if a["denoise"] else st.warning)(msg)
        left, right = st.columns([1, 1])

        if a["rule"] == "cv-compare":
            with left:
                f = go.Figure(go.Bar(
                    x=["未去噪", "去噪后"],
                    y=[a["acc_raw"], a["acc_denoised"]],
                    marker_color=["#90a4ae", "#2e7d32" if a["denoise"] else "#c62828"],
                    text=[f"{a['acc_raw']:.4f}", f"{a['acc_denoised']:.4f}"],
                    textposition="outside"))
                f.update_layout(height=320, title="交叉验证准确率对比",
                                yaxis_title="CV 准确率", yaxis_range=[0, 1.05],
                                margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(f, width="stretch")
            with right:
                st.markdown("**判据**")
                st.write(a["reason"])
                st.write(f"增量阈值 0.005：Δ={a['delta_acc']:+.4f}，"
                         f"{'超过' if a['denoise'] else '未超过'}阈值。")
        else:
            with left:
                tt = np.where(np.isfinite(tau), tau, 0.0)
                f = go.Figure(go.Bar(x=num_names, y=tt,
                                     marker_color=["#2e7d32" if v >= TAU_ON
                                                   else "#c62828" for v in tt]))
                f.add_hline(y=TAU_ON, line_dash="dash", line_color="#2e7d32",
                            annotation_text="τ̂=0.8 开去噪")
                f.add_hline(y=TAU_OFF, line_dash="dash", line_color="#c62828",
                            annotation_text="τ̂=0.5 不开")
                f.update_layout(height=320,
                                title="各数值列自相关形状比 τ̂=acf(8)/acf(1)",
                                yaxis_title="τ̂", margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(f, width="stretch")
            with right:
                st.markdown("**判据**")
                st.write(a["reason"])
                if np.isfinite(acf1).any():
                    st.write("acf(1) 中位数 %.3f，τ̂ 中位数 %s（%s）"
                             % (float(np.nanmedian(acf1)),
                                ("%.3f" % float(np.nanmedian(tau))
                                 if np.isfinite(np.nanmedian(tau)) else "n/a"),
                                a.get("tau_band", "-")))
                if a.get("sort_key") and a["sort_key"].get("blocked"):
                    st.write("排序键检验：相邻行同组 %.0f%%，随机期望 %.0f%%（%.1f×）"
                             "→ 行序由分组键决定。"
                             % (a["sort_key"]["same_adj"] * 100,
                                a["sort_key"]["expected"] * 100, a["sort_key"]["ratio"]))
                if a.get("caveat"):
                    st.warning(a["caveat"])

        st.caption("一维小波去噪假设「相邻样本构成平滑信号」。表格数据的行序通常是任意的："
                   "自相关接近 0 时沿样本维去噪抹掉的是真实个体差异，而不是噪声。"
                   "这就是为什么多数表格数据上去噪反而掉点——论文 §D1 的结论，"
                   "在真实成绩数据上表现为 CV 准确率下降约 50 个百分点。")
    else:
        st.info(f"去噪开关已手动设为 **{denoise_mode}**，未调用顾问。")

    rows = s.get("diagnostics")
    m = s.get("metrics") or {}
    if m and "ACC" in m:            # 无标签时 metrics 是 {silhouette, n_clusters}
        st.divider()
        st.subheader("二、分型质量")
        a1, a2, a3 = st.columns(3)
        a1.metric("ACC", f"{m['ACC']:.4f}")
        a2.metric("NMI", f"{m['NMI']:.4f}")
        a3.metric("ARI", f"{m['ARI']:.4f}")

        if levels:
            cnt = pd.Series(np.asarray(est.y_, int)).value_counts().sort_index()
            st.dataframe(pd.DataFrame({
                "编号": list(cnt.index),
                "对应档位": [label_name(levels, c) for c in cnt.index],
                "样本数": cnt.values,
            }), width="stretch", hide_index=True)
            st.caption("编号是标签的内部编码，看「对应档位」这一列才是有含义的。")

        left, right = st.columns([1, 1])
        with left:
            yt = np.asarray(est.y_, int)
            yp = np.asarray(est.labels_, int)
            cls = sorted(set(yt.tolist()) | set(yp.tolist()))
            cm = np.zeros((len(cls), len(cls)), int)
            pos = {c: i for i, c in enumerate(cls)}
            for t_, p_ in zip(yt, yp):
                cm[pos[int(t_)], pos[int(p_)]] += 1
            f = go.Figure(go.Heatmap(
                z=cm, x=[f"预测 {label_name(levels, c)}" for c in cls],
                y=[f"真实 {label_name(levels, c)}" for c in cls],
                colorscale="Blues", text=cm, texttemplate="%{text}",
                hovertemplate="真实 %{y} / 预测 %{x}：%{z} 人<extra></extra>"))
            f.update_layout(height=360, title="混淆矩阵（行=真实档位，列=预测档位）",
                            margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(f, width="stretch")

        with right:
            P = pca_frame(est.X_num_)
            if P is not None:
                f = go.Figure()
                for c in sorted(set(yt.tolist())):
                    msk = yt == c
                    f.add_trace(go.Scatter(
                        x=P[msk, 0], y=P[msk, 1], mode="markers",
                        name=f"真实 {label_name(levels, c)}",
                        marker=dict(size=7, opacity=0.85,
                                    symbol=["circle", "square", "diamond",
                                            "cross", "x", "triangle-up"][c % 6]),
                        text=[f"预测 {label_name(levels, p)}" for p in yp[msk]],
                        hovertemplate="%{text}<extra></extra>"))
                f.update_layout(height=360,
                                title="样本分布（PCA 前两主成分，按真实档位着色）",
                                xaxis_title="PC1", yaxis_title="PC2",
                                margin=dict(l=10, r=10, t=40, b=10),
                                legend=dict(orientation="h", y=-0.18))
                st.plotly_chart(f, width="stretch")
                st.caption("投影只用了数值列（类别列不参与 PCA）。"
                           "同一颜色的点若被分成不同形状聚集区，说明分型过度切分；"
                           "不同颜色大面积混叠，说明这几项特征不足以区分档位。")

    if rows:
        st.divider()
        st.subheader("三、特征权重与可信度")
        render_weights(est, s, rows)
    else:
        st.divider()
        render_unsupervised(res)

    return rows


def render_weights(est, s, rows):
    dg = est.diagnostics_
    if dg.get("identifiable") is False:
        st.error("**权重不可识别**：%s 所有结论按红灯处理，"
                 "此时讨论单个权重的大小没有意义。"
                 % (dg.get("unstable_reason") or "数值不稳定"))
    else:
        if dg["separable"]:
            st.warning("数据近似可分（复合似然几乎饱和）：大量权重组合等价，"
                       "权重不可辨识，以下诊断仅供参考。")
        if dg["unstable"]:
            st.error(f"Hessian 病态（cond={dg['cond_H']:.3g}），数值不稳定，结论需谨慎。")
    fdr = (s.get("fdr") or {})
    if fdr:
        st.info("**多重检验校正**：%d 个权重同时检验（逐个看 |z|≥2 会把假阳性率放大到 "
                "1−(1−α)^m），已按 BH-FDR α=%.2f 校正，显著 %d 个；"
                "下表的判定列用的是校正后档位。"
                % (fdr.get("n_tests", len(rows)), fdr.get("alpha", 0.05),
                   fdr.get("n_sig", 0)))

    names = [r["feature"] for r in rows]
    w = np.array([r["weight"] for r in rows])
    se = np.array([r["se"] for r in rows])
    zz = np.array([r["z"] for r in rows])
    colors = [COLOR[r.get("level_fdr", r["level"])] for r in rows]
    uniform = 1.0 / len(rows)

    left, right = st.columns([3, 2])
    with left:
        f = go.Figure(go.Bar(
            x=names, y=w,
            error_y=dict(type="data", array=1.96 * se, visible=True, color="#555",
                         thickness=1.4),
            marker_color=colors,
            hovertemplate="%{x}<br>权重 %{y:.4f}<extra></extra>"))
        f.add_hline(y=uniform, line_dash="dash", line_color="#888",
                    annotation_text=f"均匀权重 1/m = {uniform:.3f}")
        f.update_layout(height=400, yaxis_title="权重", xaxis_title="特征",
                        title="特征权重（误差棒为 95% 置信区间 ±1.96·SE）",
                        margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(f, width="stretch")

    with right:
        f = go.Figure()
        f.add_vrect(x0=-1, x1=1, fillcolor="#eeeeee", line_width=0, opacity=0.6,
                    layer="below")
        f.add_trace(go.Bar(x=zz, y=names, orientation="h", marker_color=colors,
                           hovertemplate="%{y}<br>z = %{x:.2f}<extra></extra>"))
        f.add_vline(x=2, line_dash="dash", line_color="#2e7d32")
        f.add_vline(x=-2, line_dash="dash", line_color="#2e7d32")
        f.add_vline(x=0, line_color="#bbb")
        f.update_layout(height=400, xaxis_title="z = (权重 − 1/m) / 标准误",
                        title="可辨识性 z 值（灰色区为 |z|<1）",
                        margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(f, width="stretch")

    need = np.array([r["n_needed"] for r in rows], float)
    plan_color = [COLOR[r.get("level_fdr", r["level"])] if r["level"] == "boundary"
                  else ("#2e7d32" if r["reachable"] else "#c62828") for r in rows]
    f = go.Figure(go.Bar(
        x=names, y=np.clip(need, 1, None),
        marker_color=plan_color,
        hovertemplate="%{x}<br>需 %{y:.0f} 样本<extra></extra>"))
    f.add_hline(y=s["n"], line_dash="dot", line_color="#333",
                annotation_text=f"当前样本 {s['n']}")
    f.update_yaxes(type="log", title_text="达标所需样本量（对数轴，目标 SE ≤ 0.02）")
    f.update_layout(height=340, title="样本量规划（CRB ∝ 1/N，灰=权重已在边界，不适用）",
                    margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(f, width="stretch")

    def verdict(r):
        """分档带上方向：z 为负表示显著低于均匀权重。"""
        base = r["level_zh"]
        if r["level"] in ("boundary", "red"):
            return base
        return base + ("（高于均匀）" if r["z"] >= 0 else "（低于均匀）")

    tbl = pd.DataFrame(rows)[["feature", "weight", "se", "z", "q", "sig",
                              "level_zh", "n_needed"]]
    tbl.columns = ["特征", "权重", "标准误", "z 值", "q 值", "显著", "判定", "达标所需样本"]
    tbl["显著"] = ["是" if v else "—" for v in tbl["显著"]]
    tbl["q 值"] = [("%.2e" % v) for v in tbl["q 值"]]
    tbl["判定"] = [verdict(r) for r in rows]
    tbl["权重"] = tbl["权重"].round(4)
    tbl["标准误"] = tbl["标准误"].round(4)
    tbl["z 值"] = tbl["z 值"].round(2)
    tbl["达标所需样本"] = tbl["达标所需样本"].astype(str)
    for i, r in enumerate(rows):
        if not r["reachable"]:
            tbl.loc[i, "达标所需样本"] = "不可达"
    st.dataframe(tbl, width="stretch", hide_index=True)

    with st.expander("判定口径说明"):
        st.markdown(
            "- **z = (权重 − 1/m) / 标准误**，衡量该特征的权重是否显著偏离"
            "「所有特征同等重要」这一零假设。\n"
            "- **q 值**：m 个权重同时检验的 BH-FDR 校正后 p 值，q ≤ α 才算显著。\n"
            "- 校正后 |z| ≥ 2 且 q ≤ α 判为**可信**（绿）；"
            "|z| ≥ 1 且 q ≤ α 为**勉强**（黄）；其余为**不可辨识**（红）。\n"
            "- **z 为负且显著，表示该特征权重显著低于均匀**，即区分度不足，"
            "不是「不重要」而是「数据没支撑它更重要」——表格「判定」列会标出方向。\n"
            "- 权重落到单纯形边界（≈0）记为**未启用(边界)**，此时该方向 Fisher "
            "信息为 0，标准误无意义，不能解读为「权重精确为 0」。\n"
            "- 标准误由 Godambe 三明治估计 H⁻¹JH⁻¹ 经 delta method 换算到权重空间得到，"
            "对模型误设定稳健；**达标所需样本**按 CRB ∝ 1/N 外推至 SE ≤ 0.02。")

    render_stability(res)

    st.divider()
    st.subheader("四、这份结果说明什么")
    render_verdict_text(est, s, rows)


def render_verdict_text(est, s, rows):
    m = (s.get("metrics") or {}) if s.get("metrics") and "ACC" in s["metrics"] else None
    a = est.advisor_
    green = [r for r in rows if r.get("level_fdr", r["level"]) == "green"]
    red = [r for r in rows if r.get("level_fdr", r["level"]) == "red"]
    amber = [r for r in rows if r.get("level_fdr", r["level"]) == "amber"]
    boundary = [r for r in rows if r["level"] == "boundary"]
    strong = sorted(green, key=lambda r: -r["z"])
    weak = sorted([r for r in rows if r["z"] <= -2], key=lambda r: r["z"])

    lines = []
    lines.append(
        f"**样本与设置**：{s['n']} 条样本、{len(rows)} 个特征、k={s['k']}；"
        f"分型质量 ACC {m['ACC']:.4f} / NMI {m['NMI']:.4f} / ARI {m['ARI']:.4f}。"
        if m else
        f"**样本与设置**：{s['n']} 条样本、{len(rows)} 个特征、k={s['k']}（无标签，"
        f"仅做聚类，不提供权重诊断）。")

    if strong:
        top = strong[0]
        lines.append(
            f"**主要驱动因素**：{top['feature']}（权重 {top['weight']:.3f}，"
            f"z={top['z']:+.2f}）是区分度最强的一项"
            + (f"，其后依次是 "
               + "、".join(f"{r['feature']}(z={r['z']:+.2f})" for r in strong[1:4])
               if len(strong) > 1 else "")
            + "。这几项的权重在当前样本量下可以被数据支撑。")
    else:
        lines.append("**主要驱动因素**：没有任何一项特征的权重达到 |z| ≥ 2，"
                     "当前样本量不足以认定任何特征主导分型。")

    if weak:
        lines.append(
            "**区分度不足**："
            + "、".join(f"{r['feature']}(z={r['z']:+.2f})" for r in weak)
            + " 的权重显著**低于**均匀水平——不是不重要，而是按现有数据它拉不开档次，"
              "是最直接的可改进抓手。")

    if red or amber:
        gap = sorted([r for r in red + amber if r["reachable"]],
                     key=lambda r: -r["n_needed"])[:3]
        if gap:
            lines.append(
                "**样本量不足**："
                + "、".join(f"{r['feature']} 需约 {r['n_needed']:,} 条" for r in gap)
                + f"（当前 {s['n']} 条）。这些特征现在给不出结论，"
                  "补样本比调参数更有效。")
        unreach = [r for r in red + amber if not r["reachable"]]
        if unreach:
            lines.append("**结构性问题**："
                         + "、".join(r["feature"] for r in unreach)
                         + " 即便补样本也达不到目标精度，说明它与分型基本无关，"
                           "建议移出特征集。")

    if boundary:
        lines.append("**未启用（单纯形边界）**："
                     + "、".join(r["feature"] for r in boundary)
                     + " 的权重压到了 0，该方向 Fisher 信息为 0，"
                       "只能说「数据没有给这个特征分配权重」，不能断言它无用；"
                       "要不要保留它，应由业务判断而非这次拟合决定。")

    if a is not None:
        verb = "启用" if a["denoise"] else "关闭"
        extra = ""
        if a["rule"] == "cv-compare":
            extra = (f"（CV 准确率 {a['acc_raw']:.4f} → {a['acc_denoised']:.4f}，"
                     f"Δ={a['delta_acc']:+.4f}）")
        lines.append(f"**去噪建议**：{verb}沿样本维小波去噪{extra}。"
                     "表格数据行序无结构时，去噪抹掉的是真实个体差异。")

    advice = []
    if strong:
        advice.append(f"优先看 {strong[0]['feature']}，它的信号最强")
    if weak:
        advice.append(f"复核 {weak[0]['feature']} 的评分或采集口径，它目前区分度不足")
    dead = [r["feature"] for r in red + amber if not r["reachable"]]
    if dead:
        advice.append("把 " + "、".join(dead) + " 移出特征集再跑一次——"
                      "补样本也达不到目标精度")
    elif red and any(r["reachable"] for r in red):
        advice.append("补样本到 "
                      f"{max(r['n_needed'] for r in red if r['reachable']):,} 条量级，"
                      "比调参数更有效")
    if boundary:
        advice.append("按业务判断决定 " + "、".join(r["feature"] for r in boundary)
                      + " 是否留在特征集（本次拟合没给它权重）")
    if advice:
        lines.append("**下一步**：" + "；".join(advice) + "。")

    for line in lines:
        st.markdown(line)

    st.caption("以上文字由诊断结果自动生成：绿/黄/红来自 z 值分档，"
               "所需样本量来自 CRB ∝ 1/N 外推。它给出的是"
               "「当前数据能支撑到什么程度」，而不是特征重要性的点估计。")

    render_export(res, df=res.get("df_used"), name=res.get("name"))


def render_export(res, df=None, name=None):
    """导出与复现：一键 HTML 报告 + 与旧快照比漂移。"""
    import json
    import os
    import tempfile

    est = res["est"]
    st.divider()
    st.subheader("导出与复现")
    st.caption("一次分析本身没有价值，对比才有。报告里带数据指纹（换没换人一眼可见）"
               "和全部复现参数；快照存下来，下次直接比：权重变了是真漂移还是噪声。")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("生成 HTML 报告", key="btn_report"):
            try:
                from hybridscope.report import build_html
                html = build_html(est, df=df, source=name or "上传数据",
                                  title="HybridScope 分析报告")
                st.download_button("下载报告（自包含 HTML）", data=html,
                                   file_name="hybridscope_report.html",
                                   mime="text/html", key="dl_report")
            except Exception as exc:
                st.error("报告生成失败：%s" % exc)
    with c2:
        old = st.file_uploader("上传旧快照 JSON，判断漂移", type=["json"],
                               key="up_snap")
        if old is not None:
            try:
                from hybridscope.monitor import compare, load_snapshot
                snap = est.snapshot(df=df, name=name or "本次")
                tmp = os.path.join(tempfile.gettempdir(), "_hs_old_snap.json")
                with open(tmp, "wb") as f:
                    f.write(old.getbuffer())
                rep = compare(load_snapshot(tmp), snap,
                              alpha=getattr(est, "fdr_alpha", 0.05))
                (st.error if rep["kind"] == "weight_drift"
                 else st.warning if rep["kind"] in ("same_data", "no_se")
                 else st.success)("**%s**" % rep["headline"])
                rows = [{"特征": r["feature"], "w_A": round(r["w_a"], 4),
                         "w_B": round(r["w_b"], 4), "Δw": round(r["delta"], 4),
                         "z": None if r["z"] is None else round(r["z"], 2),
                         "判定": r["verdict"]} for r in rep["rows"]]
                if rows:
                    st.dataframe(pd.DataFrame(rows), width="stretch",
                                 hide_index=True)
                if rep["data"].get("flagged"):
                    st.write("数据侧漂移列：%s" % "、".join(rep["data"]["flagged"]))
                if rep.get("chi2") is not None and rep.get("p_global") is not None:
                    st.caption("整体卡方 %.2f（df=%d），p=%.3f"
                               % (rep["chi2"], rep["chi2_df"], rep["p_global"]))
            except Exception as exc:
                st.error("快照对比失败：%s" % exc)

    if st.button("下载本次快照 JSON", key="btn_snap"):
        try:
            snap = est.snapshot(df=df, name=name or "本次")
            st.download_button("保存快照", data=json.dumps(
                snap, ensure_ascii=False, indent=2, default=str),
                file_name="hybridscope_snapshot.json",
                mime="application/json", key="dl_snap")
        except Exception as exc:
            st.error("快照生成失败：%s" % exc)


def render_stability(res):
    """结构可信度：分半迁移稳定性 vs 列置换零分布（不依赖标签）。"""
    est = res["est"]
    r = getattr(est, "stability_", None)
    if not r or "level" not in r:
        return
    st.divider()
    st.subheader("结构可信度：这套分型是真实结构还是聚类的顺势而为")
    (st.success if r["level"] == "green"
     else st.warning if r["level"] == "amber" else st.error)(
        "结论：%s —— %s" % (r["level_zh"], r["verdict"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("迁移一致性 ARI", "%.3f" % r["stability"])
    c2.metric("严格零分布", "%.3f" % r["null_mean"])
    c3.metric("z / p（严格）",
              "%s / %s" % ("%.2f" % r["z"] if r["z"] is not None else "n/a",
                           "%.3f" % r["p_null"] if np.isfinite(r["p_null"]) else "n/a"))
    zg, pg = r.get("z_gauss"), r.get("p_gauss")
    c4.metric("z / p（宽松）",
              "%s / %s" % ("%.2f" % zg if zg is not None else "n/a",
                           "%.3f" % pg if pg is not None and np.isfinite(pg) else "n/a"))

    left, right = st.columns([1, 1])
    with left:
        f = go.Figure()
        f.add_trace(go.Histogram(x=r["null"], nbinsx=20, marker_color="#90a4ae",
                                 opacity=0.75, name="严格（列打乱，保留边际）"))
        f.add_vline(x=r.get("gauss_mean", 0.0), line_dash="dot", line_color="#546e7a",
                    annotation_text="宽松基线 %.3f" % r.get("gauss_mean", 0.0))
        f.add_vline(x=r["stability"], line_color="#c62828", line_width=3,
                    annotation_text="实测 %.3f" % r["stability"])
        f.update_layout(height=320, title="实测稳定性在零分布中的位置",
                        xaxis_title="迁移 ARI", yaxis_title="次数",
                        margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(f, width="stretch")
    with right:
        lofo = r.get("lofo") or []
        if lofo:
            d = pd.DataFrame(lofo)
            f = go.Figure(go.Bar(
                x=d["drop"], y=d["feature"], orientation="h",
                marker_color=["#2e7d32" if v > 0 else "#c62828" for v in d["drop"]],
                hovertemplate="%{y}<br>去掉后稳定性变化 %{x:+.3f}<extra></extra>"))
            f.add_vline(x=0, line_color="#888")
            f.update_layout(height=320, title="逐特征留一：去掉后稳定性变化",
                            xaxis_title="Δ 稳定性（正=该特征在撑着分型）",
                            margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(f, width="stretch")

    st.caption("做法：随机分两半各自聚类，用加权距离把 B 半的点迁移到最近的 A 半簇标签，"
               "算迁移标签与 B 自身聚类的 ARI（机会校正，无结构时落在 0 附近）。"
               "两个零分布：严格 = 逐列独立打乱（保留每列边际分布，只抹掉列间依赖），"
               "宽松 = 同均值同方差的正态替代（连边际形状都不要）。"
               "边际本身多峰时（成绩天然分档、设备天然分工况），严格基线会偏高，"
               "所以「通过严格对照」才是最硬的证据；只通过宽松对照说明"
               "比无结构噪声强，但还可能是各列边际自身造成的团块。")


def render_unsupervised(res):
    est, s = res["est"], res["s"]
    st.subheader("二、分型结果")
    lab = np.asarray(est.labels_, int)
    sizes = pd.Series(lab).value_counts().sort_index()
    left, right = st.columns([1, 1])
    with left:
        f = go.Figure(go.Bar(x=[f"簇 {c}" for c in sizes.index], y=sizes.values,
                             marker_color="#546e7a",
                             text=sizes.values, textposition="outside"))
        f.update_layout(height=340, title="各簇规模", yaxis_title="样本数",
                        margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(f, width="stretch")
    with right:
        P = pca_frame(est.X_num_)
        if P is not None:
            f = go.Figure()
            for c in sizes.index:
                msk = lab == c
                f.add_trace(go.Scatter(x=P[msk, 0], y=P[msk, 1], mode="markers",
                                       name=f"簇 {c}", marker=dict(size=7, opacity=0.85)))
            f.update_layout(height=340, title="样本分布（PCA 前两主成分，按簇着色）",
                            xaxis_title="PC1", yaxis_title="PC2",
                            margin=dict(l=10, r=10, t=40, b=10),
                            legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(f, width="stretch")
    st.caption("无标签时没有外部标准可比对，簇数与 k 近邻图的连通性有关；"
               "簇规模严重失衡或 PCA 上看不出分离，说明这批特征未必存在自然分型。")

    render_cluster_profile(est, res)

    st.subheader("三、特征权重与可信度")
    st.info("无标签场景拿不到 Godambe 三明治标准误（复合似然需要标签定义），"
            "因此换一条不依赖标签的证据链：分半迁移稳定性 + 列置换零分布。"
            "在左侧选定标签列可额外得到标准误与红黄绿灯。")
    if est.w_ is not None:
        f = go.Figure(go.Bar(x=est.feature_names_, y=est.w_, marker_color="#546e7a"))
        f.add_hline(y=1.0 / len(est.w_), line_dash="dash", line_color="#888",
                    annotation_text="均匀权重 1/m")
        f.update_layout(height=340, yaxis_title="权重",
                        title="以轮廓系数为目标学到的特征权重（无置信区间）",
                        margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(f, width="stretch")
    render_stability(res)


def render_compare(results: list[dict]):
    """逐个分析模式：跨数据集指标 + 共有特征的权重/z 对比。"""
    st.divider()
    st.subheader("跨数据集对比")

    rows = []
    for r in results:
        s, m = r["s"], r["s"].get("metrics")
        rows.append({
            "数据集": r["name"],
            "样本数": s["n"],
            "ACC": f"{m['ACC']:.4f}" if m and "ACC" in m else "—",
            "NMI": f"{m['NMI']:.4f}" if m and "ACC" in m else "—",
            "ARI": f"{m['ARI']:.4f}" if m and "ACC" in m else "—",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    diag = [(r["name"], r["s"]["diagnostics"]) for r in results if r["s"].get("diagnostics")]
    if len(diag) < 2:
        st.info("至少两个数据集都带标签，才能对比权重。")
        return

    feats: list[str] = []
    for _, rs in diag:
        for r in rs:
            if r["feature"] not in feats:
                feats.append(r["feature"])
    shared = [f for f in feats
              if all(f in [x["feature"] for x in rs] for _, rs in diag)]
    if not shared:
        st.warning("各数据集的字段名没有重叠（字段口径不一致），无法直接对比权重。"
                   "若要比较，请先把列名统一成同一套再上传。")
        return

    dropped = [f for f in feats if f not in shared]
    if dropped:
        st.caption("以下字段只存在于部分数据集，未纳入对比："
                   + "、".join(dropped))

    left, right = st.columns([1, 1])
    with left:
        f = go.Figure()
        for name, rs in diag:
            d = {x["feature"]: x for x in rs}
            f.add_trace(go.Bar(name=name, x=shared, y=[d[x]["weight"] for x in shared]))
        f.update_layout(height=380, barmode="group", yaxis_title="权重",
                        title="权重对比（仅共有字段）",
                        margin=dict(l=10, r=10, t=40, b=10),
                        legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(f, width="stretch")
    with right:
        f = go.Figure()
        for name, rs in diag:
            d = {x["feature"]: x for x in rs}
            f.add_trace(go.Bar(name=name, x=shared, y=[d[x]["z"] for x in shared]))
        f.add_hline(y=2, line_dash="dash", line_color="#2e7d32")
        f.add_hline(y=-2, line_dash="dash", line_color="#2e7d32")
        f.update_layout(height=380, barmode="group", xaxis_title="特征",
                        yaxis_title="z 值", title="可辨识性对比",
                        margin=dict(l=10, r=10, t=40, b=10),
                        legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(f, width="stretch")

    st.caption("同一字段在不同数据集上 z 值符号相反，说明它的作用随批次变化，"
               "不能当作稳定结论；各数据集都显著同向的字段才是可靠抓手。")


with st.sidebar:
    st.header("数据")
    src = st.radio("数据源", ["内置示例", "上传文件"], horizontal=True)
    frames: dict[str, pd.DataFrame] = {}
    df = None
    mode = "single"
    merge_rep = None
    each_target = "无"

    if src == "上传文件":
        ups = st.file_uploader("选择文件（可多选）", type=["csv", "xlsx"],
                               accept_multiple_files=True)
        for up in ups or []:
            try:
                frames[up.name] = read_any(up)
            except Exception as exc:                      # 坏文件不该拖垮整页
                st.error(f"{up.name} 读取失败：{exc}")
        if len(frames) > 1:
            mode = st.radio(
                "多文件怎么处理",
                ["纵向合并（追加样本）", "横向合并（按主键对齐）", "逐个分析"],
                help="同字段多批次 → 纵向；同一样本的不同字段 → 横向；"
                     "口径不一致、只想分别看结论 → 逐个。")
            mode = {"纵向合并（追加样本）": "concat",
                    "横向合并（按主键对齐）": "merge",
                    "逐个分析": "each"}[mode]
            ov = table_overview(frames)
            if mode == "concat":
                keep_common = st.checkbox("只保留所有文件都有的列", value=True)
                add_src = st.checkbox("加入「数据来源」列", value=True)
                df, merge_rep = concat_frames(
                    frames, keep="intersection" if keep_common else "union",
                    add_source=add_src)
            elif mode == "merge":
                key = st.selectbox("主键列（两表都有的唯一标识）", ov["common"])
                how = st.selectbox("连接方式", ["inner", "left", "outer"],
                                   help="inner 只留对上的行；left 保留左表全部行")
                df, merge_rep = merge_frames(frames, key=key, how=how)
            else:
                first = ov["names"][0]
                each_target = st.selectbox(
                    "标签列（各文件按列名匹配，找不到就按无监督处理）",
                    ["无"] + list(frames[first].columns))
        elif len(frames) == 1:
            df = next(iter(frames.values()))
    else:
        kind = st.radio("示例场景", ["设备工况", "学情分型"], horizontal=True)
        n = st.slider("样本量", 120, 600, 240, 30)
        if kind == "设备工况":
            snr = st.slider("信噪比 (dB)", 0.0, 12.0, 4.0, 0.5)
            df = demo_frame(n, snr)
        else:
            df = demo_grades(n)

    st.header("参数")
    k = st.slider("k 近邻", 3, 15, 5)
    denoise_mode = st.selectbox("去噪", ["auto", "on", "off"],
                                help="auto 由去噪顾问按交叉验证决定")
    auto_prep = st.checkbox(
        "数据体检（缺失/离群/低频档自动修补，不再整行丢弃）", value=True)
    stability_on = st.checkbox("结构可信度检验（分半迁移 + 置换零分布）", value=True)
    particles = st.slider("PSO 粒子数", 8, 40, 15)
    iters = st.slider("PSO 迭代", 10, 60, 20)

    st.header("口径与算力")
    tmpl_name = st.selectbox(
        "领域模板", ["无"] + templates.names(),
        help="grades 成绩 / survey 问卷量表 / ops 设备运维。"
             "选了会自动认角色、按业务语义解读权重，survey 还可按维度合成分数。")
    speed = st.selectbox(
        "算力档位", ["off", "auto", "fast", "thorough"], index=0,
        help="auto 按样本量与特征数自动定 PSO 规模与置换次数；fast 省三到四成时间，"
             "代价是随机搜索更粗、可信度检验精度下降；off 沿用上面的手动参数。")
    sch_up = st.file_uploader("字段映射 JSON（跨批次对比时用）", type=["json"],
                              help="把不同批次的列名折叠到同一套 canonical 字段上，"
                                   "口径不一致的列会被标为不可比、不参与 Δw 判定。")
    profile = (st.text_input("本批次名（对应映射里的 profile）", value="")
               if sch_up is not None else "")
    dims_json, drop_straight = "", False
    if tmpl_name == "survey":
        dims_json = st.text_area(
            "维度定义 JSON（可留空）", value="", height=90,
            help='如 {"学习投入":["Q1","Q2","Q3"],"学习焦虑":["Q4","Q5"]}，'
                 '键 "_reverse" 里列反向计分题。留空则直接用题目建模。')
        drop_straight = st.checkbox("剔除直线作答（所有题同一选项）", value=False)
    run = st.button("运行分析", type="primary", width="stretch")

if mode == "each":
    st.subheader(f"数据概览（{len(frames)} 个文件，逐个分析）")
    st.dataframe(pd.DataFrame([{"文件": n, "行数": len(f), "列数": f.shape[1]}
                               for n, f in frames.items()]),
                 width="stretch", hide_index=True)
    if not run:
        st.info("点击左侧「运行分析」，各数据集会依次出报告，最后给跨数据集对比。")
        st.stop()

    results = []
    for name, d in frames.items():
        st.divider()
        st.header(f"数据集：{name}")
        render_overview(d)
        if len(d.dropna()) < 10:
            st.error(f"{name} 去掉缺失行后不足 10 条样本，跳过。")
            continue
        tgt = None if each_target == "无" or each_target not in d.columns else each_target
        cats = [c for c in d.columns
                if c != tgt and not pd.api.types.is_numeric_dtype(d[c])]
        if tgt is None:
            st.info("未找到指定的标签列，本数据集按无监督聚类处理。")
        with st.spinner(f"{name} 拟合中…"):
            res = analyze(d, tgt, cats, k, denoise_mode, particles, iters,
                          auto_prep=auto_prep, run_stability=stability_on,
                          speed=speed)
        res["name"] = name
        render_report(res, denoise_mode)
        results.append(res)

    render_compare(results)
    st.stop()

if df is None:
    st.info("请在左侧选择数据源。")
    st.stop()

# ---- 口径统一：字段映射折叠 + 领域模板预处理 ----
hint_target, hint_cats = None, []
if sch_up is not None:
    try:
        sch = Schema.from_dict(json.loads(sch_up.getvalue().decode("utf-8")))
        uni = sch.apply(df, profile.strip() or "batch")
        if uni.ok():
            df = uni.df
            st.success("已按字段映射折叠为 canonical 列："
                       + "、".join(str(c) for c in df.columns))
            for w in uni.warnings:
                st.caption("映射提示：" + str(w))
        else:
            st.error("字段映射应用失败："
                     + "；".join(str(e) for e in (uni.report.get("errors") or [])))
    except Exception as exc:
        st.error(f"字段映射读取失败：{exc}")

if tmpl_name != "无":
    tmpl = templates.get(tmpl_name)
    try:
        dims = {}
        if dims_json.strip():
            raw = json.loads(dims_json)
            dims = {k: v for k, v in raw.items() if not str(k).startswith("_")}
        items = sorted({c for v in dims.values() for c in v} & set(df.columns))
        if drop_straight and items:
            df, n = tmpl.drop_invalid(df, items)
            if n:
                st.caption(f"已剔除直线作答 {n} 份，余 {len(df)} 份")
        if dims:
            scores, audit = tmpl.dimension_scores(
                df, dims, reverse=list(json.loads(dims_json).get("_reverse") or []))
            df = pd.concat([df.reset_index(drop=True),
                            scores.reset_index(drop=True)], axis=1)
            st.success("已合成维度分：" + "、".join(str(c) for c in scores.columns))
            for w in audit.get("warnings") or []:
                st.warning(str(w))
            for dim, info in (audit.get("dimensions") or {}).items():
                a = info.get("alpha")
                rev = info.get("reversed") or []
                st.caption("维度 %s：%d 题%s，α=%s"
                           % (dim, info.get("n_items", 0),
                              "（含反向题 " + "、".join(rev) + "）" if rev else "",
                              "%.2f" % a if a is not None else "—"))
        roles = tmpl.detect_roles(df)
        hint_target = tmpl.pick_target(df, roles)
        _num, _cat = tmpl.feature_split(df, target=hint_target, roles=roles)
        hint_cats = [c for c in _cat if c in df.columns]
        for c in _tmpl_caveats(tmpl, tmpl.summarize(df, target=hint_target)):
            st.caption("模板提示：" + c)
    except Exception as exc:
        st.warning(f"模板预处理未生效：{exc}")

st.subheader("数据概览")
if merge_rep is not None:
    render_merge_report(merge_rep)
render_overview(df)

usable = int(len(df.dropna()))
if usable < len(df):
    st.warning(f"有 {len(df) - usable} 行含缺失格。"
               "默认开启的数据体检会逐列修补（中位数填补 / 缺失独立成档），"
               "而不是整行丢弃；关掉体检才会退回旧行为。")
if usable < 10:
    st.error("可用样本不足 10 行，无法建模。请改用「只保留所有文件都有的列」，"
             "或换成「逐个分析」。")
    st.stop()

_col_opts = (["无"] + ([hint_target] if hint_target in df.columns else [])
            + [c for c in df.columns if c != hint_target])
target = st.selectbox("标签列（选「无」则做无监督聚类）", _col_opts,
                      help="选了模板时，模板认出的标签列排在第一个")
y_col = None if target == "无" else target
rest = [c for c in df.columns if c != y_col]
_default_cats = ([c for c in hint_cats if c in rest] if hint_cats
                 else [c for c in rest
                       if not pd.api.types.is_numeric_dtype(df[c])])
cat_cols = st.multiselect("类别列（整数编码的类别列请手动勾选）", rest,
                          default=_default_cats)

if not run:
    st.info("调整左侧参数后点击「运行分析」。")
    st.stop()

with st.spinner("拟合中…"):
    res = analyze(df, y_col, cat_cols, k, denoise_mode, particles, iters,
                  auto_prep=auto_prep, run_stability=stability_on, speed=speed)
render_report(res, denoise_mode)

st.divider()
st.subheader("导出")
st.download_button("下载结果 JSON",
                   data=json.dumps(res["s"], ensure_ascii=False, indent=2, default=float),
                   file_name="hybridscope_result.json",
                   mime="application/json")
