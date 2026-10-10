# -*- coding: utf-8 -*-
"""PSO 流程图 python fig1_pso_flowchart.py"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)


def box(ax, x, y, text, w=3.0, h=0.62, fc="#DCE9F7", ec="#2F5597"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5, color="#1F3864")


def arrow(ax, x0, y0, x1, y1, label=None, style="-|>", rad=0.0, color="#2F5597"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14,
                                 lw=1.4, color=color, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x0 + x1) / 2 + 0.15, (y0 + y1) / 2, label, fontsize=8.5, color="#555555")


def main():
    fig, ax = plt.subplots(figsize=(6.2, 9.2))
    ax.set_xlim(-1.2, 6.0)
    ax.set_ylim(-0.6, 12.2)
    ax.axis("off")

    xc = 2.4
    # 步骤 (1)–(6)
    box(ax, xc, 11.4, "开始", fc="#C6E0B4", ec="#548235")
    box(ax, xc, 10.2, "(1) 形成全部混合属性数据的\n特征向量集合 Q（式15）")
    box(ax, xc, 9.0, "(2) 选择 k 个整数作为\n各个粒子的初始近邻位置")
    box(ax, xc, 7.8, "(3) 计算特征向量集合的 k 个\n最优近邻有序集合 $p_g^1$（式16）")
    box(ax, xc, 6.4, "个体最优 = 粒子位置\n全局最优 = 最优近邻有序集合")
    box(ax, xc, 5.1, "(4) 按式(14) 更新粒子\n速度与位置")
    box(ax, xc, 3.8, "(5) 选取满足需求的特征向量集合，\n取 k 个最优近邻有序集合作局部指导")
    # 判定框（菱形）
    ax.add_patch(FancyBboxPatch((xc - 1.7, 2.2), 3.4, 1.0, boxstyle="round,pad=0.05",
                                fc="#FFF2CC", ec="#BF8F00", lw=1.4))
    ax.text(xc, 2.7, "(6) 全部特征向量\n是否满足约束条件？", ha="center", va="center",
            fontsize=9.5, color="#806000")
    box(ax, xc, 1.1, "输出混合属性聚类结果", fc="#C6E0B4", ec="#548235")

    ys = [11.4, 10.2, 9.0, 7.8, 6.4, 5.1, 3.8]
    for i in range(len(ys) - 1):
        arrow(ax, xc, ys[i] - 0.31, xc, ys[i + 1] + 0.31)
    arrow(ax, xc, 3.49, xc, 3.2)      # (5) -> 判定
    arrow(ax, xc, 2.2, xc, 1.42, label="是")
    # 否 -> 回到步骤 (2)
    arrow(ax, xc + 1.7, 2.7, 4.9, 2.7, color="#C00000")
    arrow(ax, 4.9, 2.7, 4.9, 9.0, color="#C00000")
    arrow(ax, 4.9, 9.0, xc + 1.55, 9.0, label="否", color="#C00000")

    ax.set_title("图1  PSO 算法流程图", fontsize=12, pad=8)
    fig.tight_layout()
    path = os.path.join(OUT, "fig1_pso_flowchart.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    main()
