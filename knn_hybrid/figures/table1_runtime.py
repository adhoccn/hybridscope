# -*- coding: utf-8 -*-
"""聚类时间比较（单位 s）。"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "table1_runtime.csv")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.dirname(HERE))


def regenerate(sizes=(100, 200, 300, 400), seed=0):
    from knn_hybrid.baselines import EntropyKPrototypes, SMAFuzzyCMeans
    from knn_hybrid.data import add_noise, make_hybrid_dataset
    from knn_hybrid.pipeline import KNNHybridClustering

    spec = [dict(d_num=10, d_cat=10, n_clusters=3, sep=1.4, cat_confusion=0.20, seed=seed + 11),
            dict(d_num=10, d_cat=10, n_clusters=4, sep=1.2, cat_confusion=0.30, seed=seed + 12),
            dict(d_num=10, d_cat=10, n_clusters=5, sep=1.0, cat_confusion=0.35, seed=seed + 13)]
    rows = []
    rng = np.random.default_rng(seed)
    for di, sp in enumerate(spec, 1):
        base = add_noise(make_hybrid_dataset(n_samples=400, **sp), 6.0, seed=seed + 7)
        for n in sizes:
            sub = base.subset(rng.choice(base.n, n, replace=False))
            t0 = time.perf_counter()
            KNNHybridClustering(k=5, pso_particles=15, pso_iters=20, seed=seed,
                                denoise_sort_by="pca").fit(sub.X_num, sub.X_cat, sub.y)
            tp = time.perf_counter() - t0
            K = int(sub.y.max()) + 1
            t0 = time.perf_counter()
            EntropyKPrototypes(n_clusters=K, seed=seed).fit_predict(sub.X_num, sub.X_cat, sub.y)
            t3 = time.perf_counter() - t0
            t0 = time.perf_counter()
            SMAFuzzyCMeans(n_clusters=K, seed=seed).fit_predict(sub.X_num, sub.X_cat)
            t4 = time.perf_counter() - t0
            rows.append(dict(dataset=f"DS{di}", n_samples=n,
                             proposed=round(tp, 3), ref3=round(t3, 3), ref4=round(t4, 3),
                             source="regenerated"))
            print(f"  DS{di} n={n} done", flush=True)
    return pd.DataFrame(rows)


COLS = ["dataset", "n_samples", "proposed", "ref3", "ref4"]
LABELS = ["数据集", "样本数", "所提算法 /s", "文献[3] /s", "文献[4] /s"]


def save_table_png(df, path, title="表1  不同数据集下各个算法的聚类时间比较（单位 s）"):
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(df) + 1.1))
    ax.axis("off")
    show = df.copy()
    for c in ("proposed", "ref3", "ref4"):
        show[c] = show[c].map(lambda v: f"{v:.2f}")
    tbl = ax.table(cellText=show[COLS].values, colLabels=LABELS,
                   cellLoc="center", loc="upper center", bbox=[0.0, 0.0, 1.0, 0.88])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    ax.set_title(title, fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(DATA)
    print("\n=== 表1（单位 s）===")
    print(df.pivot_table(index="dataset", columns="n_samples",
                         values=["proposed", "ref3", "ref4"]).round(2).to_string())
    save_table_png(df, os.path.join(OUT, "table1_runtime_paper.png"))
    df.to_csv(os.path.join(OUT, "table1_runtime_paper.csv"), index=False,
              encoding="utf-8-sig", columns=COLS)

    if a.regenerate:
        rd = regenerate()
        print("\n=== 本次实测（单位 s）===")
        print(rd.to_string(index=False))
        rd.to_csv(os.path.join(OUT, "table1_runtime_regenerated.csv"),
                  index=False, encoding="utf-8-sig", columns=COLS)
        save_table_png(rd, os.path.join(OUT, "table1_runtime_regenerated.png"))
    print("\nsaved to", OUT)


if __name__ == "__main__":
    main()
