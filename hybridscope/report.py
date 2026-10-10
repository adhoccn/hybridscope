# -*- coding: utf-8 -*-
"""一键 HTML 报告：把一次分析封成能存档、能转发、能复现的单个文件。

报告里必须写清三件事，否则它就是一张好看的废纸：
  1. 数据是谁（hash_rows / hash_pool：换没换人、换没换行序一眼可见）；
  2. 结论怎么来的（版本、种子、k、去噪开关、FDR 水平、bootstrap/置换次数）；
  3. 接下来该干什么（每个权重一条动作，而不是一个 z 值）。
图用 matplotlib 画成内嵌 PNG，报告不依赖任何外部资源，断网也能打开。
"""

from __future__ import annotations

import base64
import datetime as _dt
import html as _html
import io as _io

import numpy as np

from hybridscope import __version__

C_GREEN, C_AMBER, C_RED, C_GREY = "#2e7d32", "#f9a825", "#c62828", "#78909c"
LEVEL_COLOR = {"green": C_GREEN, "amber": C_AMBER, "red": C_RED,
               "boundary": C_GREY, "-1": C_GREY}
LEVEL_ZH = {"green": "可信", "amber": "勉强", "red": "不可辨识",
            "boundary": "未启用(边界)"}


# ---------------------------------------------------------------- 图

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as mpl_plt
    mpl_plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    mpl_plt.rcParams["axes.unicode_minus"] = False
    return mpl_plt


def _embed(fig) -> str:
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def fig_weights(est) -> str | None:
    if getattr(est, "w_", None) is None:
        return None
    plt = _mpl()
    w = np.asarray(est.w_, float)
    names = [str(x) for x in getattr(est, "feature_names_", range(w.size))]
    dg = getattr(est, "diagnostics_", None) or {}
    se = dg.get("se_w")
    se = np.asarray(se, float) if se is not None and np.asarray(se).size == w.size else None
    rows = (dg.get("rows") or [])
    def _lvl(i):
        if i < len(rows):
            return rows[i].get("level_fdr") or rows[i].get("level")
        return None
    colors = [LEVEL_COLOR.get(_lvl(i), C_GREY) for i in range(w.size)]
    fig, ax = plt.subplots(figsize=(max(5.0, 0.55 * w.size + 3.0), 3.4))
    yerr = se if se is not None else None
    ax.bar(names, w, color=colors, yerr=yerr, ecolor="#455a64",
           capsize=3, error_kw={"linewidth": 1.0})
    ax.axhline(1.0 / w.size, ls="--", c="#888", lw=1,
               label="均匀权重 1/m = %.3f" % (1.0 / w.size))
    ax.set_ylabel("权重")
    ax.set_title("特征权重%s" % ("（误差棒为标准误）" if se is not None
                              else "（无标准误：无标签场景）"))
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return _embed(fig)


def fig_stability(est) -> str | None:
    st = getattr(est, "stability_", None)
    if not st or "null" not in st:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    null = np.asarray(st.get("null") or [], float)
    gauss = np.asarray(st.get("gauss") or [], float)
    bins = 18
    if gauss.size:
        ax.hist(gauss, bins=bins, color="#bbdefb", alpha=0.85,
                label="宽松对照（同均值方差正态）")
    if null.size:
        ax.hist(null, bins=bins, color="#90a4ae", alpha=0.75,
                label="严格对照（逐列打乱，保留边际）")
    if null.size or gauss.size:
        ax.axvline(st.get("stability", np.nan), color=C_RED, lw=2.2,
                   label="实测 %.3f" % st.get("stability", float("nan")))
    ax.set_xlabel("分半迁移 ARI")
    ax.set_ylabel("次数")
    ax.set_title("实测结构强度 vs 零分布")
    ax.legend(fontsize=7.5, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return _embed(fig)


def fig_clusters(est) -> str | None:
    lab = getattr(est, "labels_", None)
    if lab is None:
        return None
    _, cnt = np.unique(np.asarray(lab), return_counts=True)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar([f"簇 {i}" for i in range(len(cnt))], cnt, color="#546e7a")
    ax.set_ylabel("人数 / 样本数")
    ax.set_title("分型规模（%d 簇）" % len(cnt))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return _embed(fig)


# ---------------------------------------------------------------- 组装

def _esc(x) -> str:
    return _html.escape("" if x is None else str(x))


def _fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "n/a"
        return ("%%.%df" % nd) % v
    return str(v)


def _table(headers, rows, aligns=None):
    aligns = aligns or ["left"] * len(headers)
    th = "".join('<th style="text-align:%s">%s</th>' % (a, _esc(h))
                 for a, h in zip(aligns, headers))
    tr = []
    for r in rows:
        tr.append("<tr>" + "".join(
            '<td style="text-align:%s">%s</td>' % (a, _esc(c))
            for a, c in zip(aligns, r)) + "</tr>")
    return ('<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>'
            % (th, "".join(tr)))


def build_html(est, df=None, title: str | None = None, source: str | None = None,
               template=None, template_explain: list[str] | None = None,
               notes: list[str] | None = None) -> str:
    from hybridscope.io import data_fingerprint
    from hybridscope.stability import to_text as stab_text

    s = est.summary()
    fp = None
    if df is not None:
        try:
            fp = data_fingerprint(df)
        except Exception:
            fp = getattr(est, "data_fingerprint_", None)
    if fp is None:
        fp = getattr(est, "data_fingerprint_", None)

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    title = title or "HybridScope 分析报告"
    parts = []
    A = parts.append

    A("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>")
    A("<title>%s</title><style>%s</style></head><body>" % (_esc(title), _CSS))
    A("<h1>%s</h1>" % _esc(title))
    A("<p class='sub'>生成时间 %s · HybridScope v%s%s</p>"
      % (now, __version__, (" · 数据源 %s" % _esc(source)) if source else ""))

    # ---------- 0. 复现信息
    A("<h2>零、复现信息</h2>")
    cfg = [["样本量 n", s.get("n")], ["近邻 k", s.get("k")],
           ["去噪开关", "开" if s.get("denoise") else ("关" if s.get("denoise") is False else "—")],
           ["随机种子", getattr(est, "seed", None)],
           ["PSO 粒子/迭代", "%s / %s" % (getattr(est, "pso_particles", "—"),
                                       getattr(est, "pso_iters", "—"))],
           ["FDR 水平 α", getattr(est, "fdr_alpha", None)],
           ["梯度 bootstrap", getattr(est, "n_boot", None)],
           ["稳定性 boot/置换", "%s / %s" % (getattr(est, "stab_boot", "—"),
                                         getattr(est, "stab_perm", "—"))]]
    if fp:
        cfg += [["数据指纹（含行序）", fp.get("hash_rows")],
                ["数据指纹（忽略行序）", fp.get("hash_pool")],
                ["行列数", "%d 行 × %d 列" % (fp.get("n_rows", 0), fp.get("n_cols", 0))]]
    A(_table(["项", "值"], [[a, _fmt(b, 6) if isinstance(b, float) else b] for a, b in cfg],
             ["left", "left"]))
    if fp:
        A("<p class='note'>两份报告若 hash_rows 相同即同一批数据且行序一致；"
          "hash_pool 相同而行序不同，说明同一批人换了存放顺序。"
          "把它写进报告是为了让「复现」可被核对，而不是靠口头保证。</p>")

    # ---------- 1. 数据体检
    prep = s.get("prep")
    if prep:
        A("<h2>一、数据体检</h2>")
        A("<p>%d 行 / 重复行 %d 条%s</p>"
          % (prep.get("n_rows_in", 0), prep.get("n_dup_rows_dropped", 0),
             ("，沿用整行丢弃缺失会损失 <b>%d</b> 行" % prep["dropna_would_drop"])
             if prep.get("dropna_would_drop") else ""))
        rows = [["剔除", d.get("column"), d.get("reason")]
                for d in prep.get("dropped", [])]
        rows += [["修补", a.get("column"), a.get("action")]
                 for a in prep.get("actions", [])]
        if rows:
            A(_table(["类型", "列", "处理"], rows))
        else:
            A("<p>未发现需要处理的问题。</p>")

    # ---------- 2. 去噪顾问
    adv = s.get("advisor")
    if adv:
        A("<h2>二、去噪顾问</h2>")
        A("<p><b>结论：%s去噪</b> —— %s</p>"
          % ("开" if s.get("denoise") else "不", _esc(adv.get("reason"))))
        t = [["acf(1) 中位数", _fmt(adv.get("acf1_median"), 3)],
             ["τ̂ = acf(8)/acf(1)", _fmt(adv.get("tau_median"), 3)],
             ["τ̂ 区间", adv.get("tau_band")],
             ["CV 准确率差", _fmt(adv.get("delta_acc"), 4)]]
        A(_table(["判据", "值"], t))
        if adv.get("caveat"):
            A("<p class='warn'>%s</p>" % _esc(adv["caveat"]))

    # ---------- 3. 分型质量
    A("<h2>三、分型质量</h2>")
    met = s.get("metrics") or {}
    if met:
        A(_table(["指标", "值"],
                 [[k, _fmt(v, 4) if isinstance(v, (int, float, np.floating)) else v]
                  for k, v in met.items()]))
    img = fig_clusters(est)
    if img:
        A("<img src='%s'>" % img)

    # ---------- 4. 权重与可信度
    A("<h2>四、特征权重与可信度</h2>")
    img = fig_weights(est)
    if img:
        A("<img src='%s'>" % img)
    rows = s.get("diagnostics")
    if rows:
        fdr = s.get("fdr") or {}
        A("<p>%d 个权重同时检验，BH-FDR α=%s 校正后显著 <b>%d</b> 个。"
          "逐项看 |z|≥2 会把假阳性率放大到 1−(1−α)^m，所以判定列按校正后档位给出。</p>"
          % (fdr.get("n_tests", len(rows)), fdr.get("alpha", 0.05), fdr.get("n_sig", 0)))
        if s.get("identifiable") is False:
            A("<p class='warn'>权重不可识别：%s。所有结论按红灯处理。</p>"
              % _esc(s.get("unstable_reason") or "数值不稳定"))
        tbl = [[r["feature"], _fmt(r["weight"]), _fmt(r["se"]), _fmt(r["z"], 2),
                _fmt(r.get("q"), 3), "是" if r.get("sig") else "—",
                LEVEL_ZH.get(r.get("level_fdr"), r.get("level_fdr")),
                str(r["n_needed"]) if r.get("reachable") else "不可达"]
               for r in rows]
        A(_table(["特征", "权重", "标准误", "z", "q", "显著", "判定", "达标样本"],
                 tbl, ["left"] + ["right"] * 5 + ["left", "right"]))
    elif s.get("weights"):
        A("<p>无标签场景：权重以轮廓系数为目标拟合，没有 Godambe 三明治标准误，"
          "可信度见下一节。</p>")
        A(_table(["特征", "权重"],
                 [[k, _fmt(v)] for k, v in s["weights"].items()], ["left", "right"]))

    # ---------- 5. 结构可信度
    st = getattr(est, "stability_", None)
    if st and "level" in st:
        A("<h2>五、结构可信度（不依赖标签）</h2>")
        img = fig_stability(est)
        if img:
            A("<img src='%s'>" % img)
        t = [["迁移一致性 ARI", "%.3f ± %.3f（%d 次分半）"
              % (st["stability"], st["stability_sd"], st["n_boot"])],
             ["严格对照（列打乱）", "%.3f ± %.3f → z=%s，p=%s"
              % (st["null_mean"], st["null_sd"],
                 "%.2f" % st["z"] if st.get("z") is not None else "n/a",
                 "%.3f" % st["p_null"] if np.isfinite(st.get("p_null", np.nan)) else "n/a")],
             ["宽松对照（正态替代）", "%.3f → p=%s"
              % (st.get("gauss_mean", float("nan")),
                 "%.3f" % st["p_gauss"] if np.isfinite(st.get("p_gauss", np.nan)) else "n/a")],
             ["簇数 / 轮廓系数", "%d / %s" % (st["n_clusters"], _fmt(st["silhouette"], 3))],
             ["判定", "%s（%s）" % (st["level"], st["level_zh"])]]
        A(_table(["项", "值"], t))
        A("<p>%s</p>" % _esc(st.get("verdict")))
        if st.get("top_features"):
            A("<p>撑着分型的特征：%s</p>" % _esc("、".join(st["top_features"])))
        if st.get("lofo"):
            A(_table(["去掉的特征", "权重", "去掉后 ARI", "Δ"],
                     [[r["feature"], _fmt(r["weight"]), _fmt(r["stability_without"], 3),
                       ("%+.3f" % r["drop"]) if np.isfinite(r.get("drop", np.nan)) else "n/a"]
                      for r in st["lofo"]], ["left"] + ["right"] * 3))

    # ---------- 6. 解读与动作
    A("<h2>六、解读与下一步</h2>")
    acts = action_list(est)
    A("<ul class='acts'>" + "".join(
        "<li class='%s'>%s</li>" % (lv, _esc(txt)) for lv, txt in acts) + "</ul>")
    if template_explain:
        A("<h3>领域解读</h3><ul>" + "".join(
            "<li>%s</li>" % _esc(x) for x in template_explain) + "</ul>")
    if template is not None and rows:
        try:
            for e in template.explain(rows):
                A("<p>· %s</p>" % _esc(e["text"]))
        except Exception:
            pass
    if notes:
        A("<h3>备注</h3><ul>" + "".join(
            "<li>%s</li>" % _esc(x) for x in notes) + "</ul>")

    A("<p class='foot'>本报告由 HybridScope v%s 生成。结论的强度取决于标准误，"
      "不取决于权重的大小；权重差要跟标准误比，不要跟零比。</p>" % __version__)
    A("</body></html>")
    return "\n".join(parts)


def action_list(est) -> list[tuple[str, str]]:
    """每个权重给一条动作，而不是一个 z 值。"""
    out = []
    s = est.summary()
    rows = s.get("diagnostics")
    st = getattr(est, "stability_", None)

    if s.get("identifiable") is False:
        out.append(("red", "权重整体不可识别（%s）：先解决数据与模型设定，"
                           "再谈单个权重。此时讨论谁重要没有意义。"
                    % (s.get("unstable_reason") or "数值不稳定")))
    if st and "level" in st:
        if st["level"] == "red":
            out.append(("red", "分型结构未通过对照检验：不要基于这批分型给个体贴标签，"
                               "先补样本或换特征。"))
        elif st["level"] == "amber":
            out.append(("amber", "结构只通过宽松对照：可以做探索性分组，"
                                 "但不足以支撑个体层面的结论。"))
        else:
            out.append(("green", "结构通过严格对照：分型反映的是真实结构，可以据此下结论。"))

    if rows:
        for r in rows:
            lv = r.get("level_fdr") or r.get("level")
            nm = r["feature"]
            if r.get("level") == "boundary":
                out.append(("amber", "%s：权重落在单纯形边界（%.4f），等价于它没参与分型 —— "
                                     "要么删掉这个特征，要么单独检查它的量纲与缺失。"
                           % (nm, r["weight"])))
            elif lv == "green":
                out.append(("green", "%s：权重 %.3f 显著偏离均匀（z=%.2f，q=%.3f），"
                                     "可以写进结论。" % (nm, r["weight"], r["z"], r.get("q", float("nan")))))
            elif lv == "amber":
                out.append(("amber", "%s：方向可信但精度不足（|z|=%.2f）。"
                                     "补到约 %s 个样本再下结论。"
                            % (nm, abs(r["z"]),
                               r["n_needed"] if r.get("reachable") else "不可达（换特征）")))
            else:
                out.append(("red", "%s：当前样本量下与「各特征等权」区分不开"
                                   "（|z|=%.2f）。别据此增减它的比重。"
                            % (nm, abs(r["z"]))))
    elif s.get("weights"):
        out.append(("grey", "无标签场景：权重只描述拟合结果，没有标准误，"
                            "不能直接当作「重要性」使用。"))
    return out


def write_report(path: str, est, **kw) -> str:
    """生成 HTML 报告并写盘，返回路径。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_html(est, **kw))
    return path


_CSS = """
body{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
 max-width:960px;margin:32px auto;padding:0 20px;color:#1f2933;line-height:1.65}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;margin:28px 0 8px;padding-bottom:4px;border-bottom:2px solid #e4e7eb}
h3{font-size:15px;margin:18px 0 6px}
p{margin:6px 0}
.sub{color:#616e7c;font-size:13px;margin-bottom:18px}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13px}
th,td{border:1px solid #e4e7eb;padding:6px 9px}
th{background:#f5f7fa;text-align:left;font-weight:600}
tbody tr:nth-child(even){background:#fafbfc}
img{max-width:100%;margin:10px 0;border:1px solid #e4e7eb;border-radius:4px}
.note{color:#616e7c;font-size:12.5px}
.warn{background:#fff8e1;border-left:3px solid #f9a825;padding:8px 12px;margin:10px 0}
.acts{list-style:none;padding:0}
.acts li{margin:6px 0;padding:8px 12px;border-left:4px solid #78909c;background:#f5f7fa}
.acts li.green{border-color:#2e7d32;background:#e8f5e9}
.acts li.amber{border-color:#f9a825;background:#fff8e1}
.acts li.red{border-color:#c62828;background:#ffebee}
.foot{margin-top:28px;padding-top:10px;border-top:1px solid #e4e7eb;
 color:#616e7c;font-size:12.5px}
"""
