# -*- coding: utf-8 -*-
"""聚类性能比较（(a) DS1 (b) DS2 (c) DS3）。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "fig3_accuracy.csv")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.dirname(HERE))

ATTRS = [1, 2, 3, 4]
ALGOS = [("proposed", "所提算法", "#2E7D32"),
         ("ref3", "文献[3]算法", "#C0504D"),
         ("ref4", "文献[4]算法", "#4F81BD")]


def regenerate(dims=(4, 8, 12, 16), seed=0):
    """论文与两个对比算法重跑"""
    from knn_hybrid.baselines import EntropyKPrototypes, SMAFuzzyCMeans
    from knn_hybrid.data import add_noise, make_hybrid_dataset
    from knn_hybrid.metrics import all_metrics
    from knn_hybrid.pipeline import KNNHybridClustering

    spec = [dict(d_num=10, d_cat=10, n_clusters=3, sep=1.4, cat_confusion=0.20, seed=seed + 11),
            dict(d_num=10, d_cat=10, n_clusters=4, sep=1.2, cat_confusion=0.30, seed=seed + 12),
            dict(d_num=10, d_cat=10, n_clusters=5, sep=1.0, cat_confusion=0.35, seed=seed + 13)]
    rows = []
    for di, sp in enumerate(spec, 1):
        d = add_noise(make_hybrid_dataset(n_samples=400, **sp), 6.0, seed=seed + 7)
        for ai, dt in enumerate(dims, 1):
            nn, nc = min(dt // 2, d.d_num), min(dt - dt // 2, d.d_cat)
            sub = type(d)(d.X_num[:, :nn], d.X_cat[:, :nc], d.y, f"DS{di}")
            m = KNNHybridClustering(k=5, pso_particles=15, pso_iters=20, seed=seed,
                                    denoise_sort_by="pca").fit(sub.X_num, sub.X_cat, sub.y)
            K = int(sub.y.max()) + 1
            e3 = EntropyKPrototypes(n_clusters=K, seed=seed).fit_predict(sub.X_num, sub.X_cat, sub.y)
            e4 = SMAFuzzyCMeans(n_clusters=K, seed=seed).fit_predict(sub.X_num, sub.X_cat)
            rows.append(dict(dataset=f"DS{di}", attr_num=ai,
                             proposed=m.score()["ACC"],
                             ref3=all_metrics(sub.y, e3)["ACC"],
                             ref4=all_metrics(sub.y, e4)["ACC"], source="regenerated"))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true")
    a = ap.parse_args()

    df = regenerate() if a.regenerate else pd.read_csv(DATA)

    ds_names = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(ds_names), figsize=(11.5, 4.0), sharey=True)
    width = 0.26
    for ax, name in zip(np.atleast_1d(axes), ds_names):
        sub = df[df["dataset"] == name].sort_values("attr_num")
        x = np.arange(len(sub))
        for i, (col, lab, color) in enumerate(ALGOS):
            ax.bar(x + (i - 1) * width, sub[col].to_numpy(), width, label=lab, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([f"数据属性{int(v)}" for v in sub["attr_num"]], fontsize=9)
        ax.set_title(f"({chr(97 + ds_names.index(name))}) {name}")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3, ls=":")
        ax.set_xlabel("")
    np.atleast_1d(axes)[0].set_ylabel("聚类准确率")
    np.atleast_1d(axes)[-1].legend(fontsize=9)
    fig.suptitle("图3  不同数据集下各个算法的聚类性能比较", fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT, "fig3_accuracy_bars.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("saved", path)
    print(df.groupby("dataset")[["proposed", "ref3", "ref4"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
