# -*- coding: utf-8 -*-
"""生成学情分型案例的权重图与 Markdown 报告。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hybridscope.estimator import HybridScopeEstimator
from run_case import band, merge_tail

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

HEX = {"green": "#2e7d32", "amber": "#f9a825", "red": "#c62828", "boundary": "#9e9e9e"}
ZH = {"green": "可信", "amber": "勉强", "red": "不可辨识", "boundary": "边界"}


def fit_case(df, num_cols, cat_col, target, k=5):
    work = df.copy()
    work["专业"] = merge_tail(work[cat_col].fillna("未标注"))
    y = pd.Categorical(band(target)).codes
    X_num = work[num_cols].to_numpy(float)
    X_cat = pd.Categorical(work["专业"]).codes.reshape(-1, 1)
    est = HybridScopeEstimator(k=min(k, len(work) - 1), seed=0)
    est.fit_hybrid(X_num, X_cat, y, feature_names=num_cols + ["专业"])
    return est, work


def draw(est, title, out):
    rows = est.diagnostics_["rows"]
    names = [r["feature"] for r in rows]
    w = np.array([r["weight"] for r in rows])
    se = np.array([r["se"] for r in rows])
    col = [HEX[r["level"]] for r in rows]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=160)
    ax.bar(x, w, yerr=se, capsize=3, color=col,
           error_kw=dict(ecolor="#555", lw=1))
    ax.axhline(1 / len(names), ls="--", c="#888", lw=1)
    ax.text(len(names) - 0.4, 1 / len(names), "均匀权重 1/m",
            ha="right", va="bottom", fontsize=9, color="#666")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("权重")
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, max(w.max() + se.max() + 0.08, 0.25))
    for i, r in enumerate(rows):
        ax.text(i, w[i] + se[i] + 0.012, ZH[r["level"]],
                ha="center", fontsize=8, color="#444")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def table_md(rows):
    out = ["| 特征 | 权重 | 标准误 | z | 判定 | 达标所需样本 |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        out.append("| %s | %.4f | %.4f | %+.2f | %s | %s |"
                   % (r["feature"], r["weight"], r["se"], r["z"], r["level_zh"],
                      str(r["n_needed"]) if r["reachable"] else "不可达"))
    return "\n".join(out)


def main():
    spring = pd.read_csv(DATA / "grades_2026spring.csv")
    autumn = pd.read_csv(DATA / "grades_2025autumn.csv")
    sc = ["测验1", "测验2", "测验3", "测验4", "报告1", "报告2", "报告3", "报告4"]
    sp = spring[spring[sc].sum(axis=1) > 0].reset_index(drop=True)
    ac = ["平时作业", "大作业"]
    au = autumn[autumn[ac].sum(axis=1) > 0].reset_index(drop=True)

    eA, wA = fit_case(sp, sc, "专业", sp["总评"])
    eB, wB = fit_case(au, ac, "专业", au["总评"])
    draw(eA, "2026春《人工智能导论》特征权重与可估计性（n=%d）" % len(wA),
         HERE / "fig_2026spring_weights.png")
    draw(eB, "2025秋《人工智能通识》特征权重与可估计性（n=%d）" % len(wB),
         HERE / "fig_2025autumn_weights.png")

    sA, sB = eA.summary(), eB.summary()
    aA, aB = eA.advisor_, eB.advisor_
    md = f"""# 学情分型案例：HybridScope 在真实成绩数据上的应用

数据来自两个学期的《人工智能》类课程成绩表。**已脱敏**：姓名不落盘，学号经
SHA256 单向哈希取前 10 位；原始成绩文件不进入版本库。

## 建模设置

| 项 | 案例 A | 案例 B |
|---|---|---|
| 课程 | 人工智能导论（选修，100514T003） | 人工智能通识·信息类（必修，102000C001） |
| 学期 | 2026 春 | 2025 秋 |
| 样本 | {len(wA)} 人（原始 173，剔除 1 名全零缺考） | {len(wB)} 人 |
| 数值特征 | 阶段性测验 1–4、实验报告 1–4（共 8） | 平时作业、大作业（共 2） |
| 类别特征 | 专业（长尾合并为「其他」后 {wA['专业'].nunique()} 类） | 专业（{wB['专业'].nunique()} 类） |
| 分型标签 | 总评 <70 需关注 / 70–84 合格 / ≥85 优秀 | 同左 |

标签由总评成绩切分，构造特征时**只使用原始分项**，不使用加权后的过程成绩、
实验成绩与总评，避免标签泄漏。

## 一、去噪顾问：真实数据上判定「不要去噪」

| 案例 | 未去噪 CV 准确率 | 去噪后 CV 准确率 | Δ | 结论 |
|---|---|---|---|---|
| A | {aA['acc_raw']:.4f} | {aA['acc_denoised']:.4f} | {aA['delta_acc']:+.4f} | 关闭 |
| B | {aB['acc_raw']:.4f} | {aB['acc_denoised']:.4f} | {aB['delta_acc']:+.4f} | 关闭 |

两个案例中沿样本维的小波去噪都使交叉验证准确率**大幅下降**（案例 A 下降
{abs(aA['delta_acc'])*100:.1f} 个百分点）。这与论文关于「表格数据行序不构成平滑
信号」的结论一致：学生名册顺序是任意的，沿该方向做小波阈值去噪抹掉的是真实
的个体差异，而非测量噪声。产品据此自动关闭去噪，而不是默认开启。

## 二、案例 A：2026 春《人工智能导论》

分型质量：**ACC {sA['metrics']['ACC']:.4f}　NMI {sA['metrics']['NMI']:.4f}　ARI {sA['metrics']['ARI']:.4f}**
（可分={eA.diagnostics_['separable']}，Hessian 病态={eA.diagnostics_['unstable']}）

![权重诊断](fig_2026spring_weights.png)

{table_md(eA.diagnostics_['rows'])}

**教学解读**

- **阶段性测验 4** 权重 {[r['weight'] for r in eA.diagnostics_['rows'] if r['feature']=='测验4'][0]:.3f}，
  显著高于均匀权重（z=+{[r['z'] for r in eA.diagnostics_['rows'] if r['feature']=='测验4'][0]:.2f}），
  是可辨识的：这一次测验对区分学生水平贡献最大。
- **实验报告 3** 权重显著**低于**均匀权重（z={[r['z'] for r in eA.diagnostics_['rows'] if r['feature']=='报告3'][0]:.2f}），
  方向为负且可辨识，说明该次报告区分度不足（评分过于集中），是教学改进的直接抓手。
- 其余 6 项（测验 1–3、报告 1/2/4）在当前样本量下**不可辨识**——不能据此断言
  它们不重要，只能说数据还不足以支撑结论。
- **专业**这一类别特征权重落在单纯形边界（0），说明专业背景不驱动成绩分型。

## 三、案例 B：2025 秋《人工智能通识》

分型质量：**ACC {sB['metrics']['ACC']:.4f}　NMI {sB['metrics']['NMI']:.4f}　ARI {sB['metrics']['ARI']:.4f}**

![权重诊断](fig_2025autumn_weights.png)

{table_md(eB.diagnostics_['rows'])}

平时作业权重 {[r['weight'] for r in eB.diagnostics_['rows'] if r['feature']=='平时作业'][0]:.3f}
可辨识（z={[r['z'] for r in eB.diagnostics_['rows'] if r['feature']=='平时作业'][0]:+.2f}），
大作业勉强（z={[r['z'] for r in eB.diagnostics_['rows'] if r['feature']=='大作业'][0]:+.2f}）。
特征维度降到 2 维后 Hessian 更病态（cond={eB.diagnostics_['cond_H']:.3g}），诊断置信度下降。

## 四、这个案例证明了什么

1. **方法可落地**：直接读入教务导出的 PDF/Excel 成绩表即可建模，无需手工整理。
2. **去噪结论在真实数据上成立**：论文关于「表格数据不该盲目去噪」的判断，在
   两份独立真实数据上都得到验证，且效应量很大。
3. **可估计性诊断有实际价值**：它把「哪些考核项真正区分学生」和「数据够不够」
   分开回答——前者给出改进教学的抓手，后者阻止过度解读。这是普通聚类工具给不了的。

## 复现

```bash
python cases/learning_analytics/parse_grades.py
python cases/learning_analytics/run_case.py
python cases/learning_analytics/make_report.py
```
"""
    (HERE / "REPORT.md").write_text(md, encoding="utf-8")
    print("wrote REPORT.md, fig_2026spring_weights.png, fig_2025autumn_weights.png")


if __name__ == "__main__":
    main()
