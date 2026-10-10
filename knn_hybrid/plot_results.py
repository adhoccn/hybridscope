# -*- coding: utf-8 -*-


from __future__ import annotations

import os

import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    den = pd.read_csv(os.path.join(RES, "denoise_snr.csv"))
    acc = pd.read_csv(os.path.join(RES, "accuracy_vs_dim.csv"))
    tim = pd.read_csv(os.path.join(RES, "time_vs_n.csv"))

# 去噪 SNR----
    piv = den.pivot_table(index=["数据类型", "数据集"], columns="方法", values="SNR_dB")
    tags = ["平滑采样 DS1", "表格式 DS1"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, tag in zip(axes, tags):
        sub = piv.xs(tag, level=0)
        sub.T.plot(kind="bar", ax=ax, rot=18, width=0.75)
        ax.set_title(f"去噪信噪比对比（{tag}）")
        ax.set_ylabel("SNR (dB)")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_denoise_snr.png"), dpi=130)
    plt.close(fig)

# 准确率 vs 属性维度----
    ds_names = list(acc["数据集"].unique())
    fig, axes = plt.subplots(1, len(ds_names), figsize=(13, 3.9), sharey=True)
    for ax, name in zip(axes, ds_names):
        sub = acc[acc["数据集"] == name]
        for algo, g in sub.groupby("算法"):
            g = g.sort_values("属性维度")
            ax.plot(g["属性维度"], g["ACC"], marker="o", ms=4, label=algo)
        ax.set_title(f"{name}（属性数递增）")
        ax.set_xlabel("属性维度")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("聚类准确率 ACC")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_accuracy_vs_dim.png"), dpi=130)
    plt.close(fig)

# 聚类时间 vs 样本数----
    fig, axes = plt.subplots(1, len(ds_names), figsize=(13, 3.9), sharey=True)
    for ax, name in zip(axes, ds_names):
        sub = tim[tim["数据集"] == name]
        for algo, g in sub.groupby("算法"):
            g = g.sort_values("样本数")
            ax.plot(g["样本数"], g["聚类时间s"], marker="s", ms=4, label=algo)
        ax.set_title(f"{name}（样本数递增）")
        ax.set_xlabel("样本数")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("聚类时间 (s)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_time_vs_n.png"), dpi=130)
    plt.close(fig)

    print("figures saved to", RES)


if __name__ == "__main__":
    main()
