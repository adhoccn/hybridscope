# -*- coding: utf-8 -*-
"""SNR ACC 不同样本规模下的聚类时间"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knn_hybrid.baselines import EntropyKPrototypes, SMAFuzzyCMeans
from knn_hybrid.data import add_noise, make_hybrid_dataset, make_smooth_hybrid_dataset
from knn_hybrid.denoise import denoise_numeric, snr_db
from knn_hybrid.metrics import all_metrics
from knn_hybrid.pipeline import KNNHybridClustering

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES, exist_ok=True)


def build_datasets(seed: int = 0):
    """测试数据集"""
    specs = [
        dict(d_num=10, d_cat=10, n_clusters=3, sep=1.4, cat_confusion=0.20, seed=seed + 11),
        dict(d_num=10, d_cat=10, n_clusters=4, sep=1.2, cat_confusion=0.30, seed=seed + 12),
        dict(d_num=10, d_cat=10, n_clusters=5, sep=1.0, cat_confusion=0.35, seed=seed + 13),
    ]
    out = []
    for i, s in enumerate(specs, 1):
        d = make_hybrid_dataset(n_samples=400, **s)
        d.name = f"DS{i}"
        out.append(d)
    return out


def take_dim(data, d_total: int):
    """取前 d_total 个属性"""
    n_num = d_total // 2
    n_cat = d_total - n_num
    n_num = min(n_num, data.d_num)
    n_cat = min(n_cat, data.d_cat)
    from knn_hybrid.data import HybridData
    return HybridData(data.X_num[:, :n_num], data.X_cat[:, :n_cat], data.y,
                      f"{data.name}-d{n_num + n_cat}",
                      data.num_names[:n_num], data.cat_names[:n_cat])


# 去噪 SNR
def exp1_denoise(seed=0):
    rows = []
    cases = []
    for i in range(3):
        d = make_hybrid_dataset(n_samples=300, d_num=6, d_cat=4, n_clusters=3,
                                sep=1.3, cat_confusion=0.25, seed=seed + 21 + i)
        cases.append((f"表格式 DS{i+1}", d, add_noise(d, 6.0, seed=seed + 31 + i)))
    for i in range(3):
        clean, noisy = make_smooth_hybrid_dataset(n_samples=300, d_num=6, d_cat=4,
                                                  n_clusters=3, snr_db=6.0,
                                                  seed=seed + 41 + i)
        noisy.name = f"smooth-noisy{i+1}"
        cases.append((f"平滑采样 DS{i+1}", clean, noisy))

    for tag, clean, noisy in cases:
        base = snr_db(clean.X_num, noisy.X_num)
        variants = {
            "含噪输入": noisy.X_num,
            "经典小波(通用阈值)": denoise_numeric(noisy.X_num, max_shift=0,
                                                  threshold_rule="universal"),
            "原始样本": denoise_numeric(noisy.X_num, max_shift=20,
                                          threshold_rule="bayes", sort_by="none"),
            "PCA重排序改进": denoise_numeric(noisy.X_num, max_shift=20,
                                            threshold_rule="bayes", sort_by="pca"),
        }
        for name, Xd in variants.items():
            rows.append(dict(数据类型=tag, 数据集=noisy.name, 方法=name,
                             SNR_dB=round(snr_db(clean.X_num, Xd), 3),
                             SNR增益_dB=round(snr_db(clean.X_num, Xd) - base, 3)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "denoise_snr.csv"), index=False, encoding="utf-8-sig")
    return df


#准确率 vs 属性维度
def _run_all(data, k=5, pso_p=15, pso_t=20, seed=0):
    res = {}
    t0 = time.perf_counter()
    m = KNNHybridClustering(k=k, pso_particles=pso_p, pso_iters=pso_t,
                            seed=seed, denoise_sort_by="pca").fit(data.X_num, data.X_cat, data.y)
    res["本文算法"] = (m.score(), time.perf_counter() - t0, m.labels_)

    t0 = time.perf_counter()
    mf = KNNHybridClustering(k=k, pso_particles=8, pso_iters=10,
                             seed=seed, denoise_sort_by="pca").fit(data.X_num, data.X_cat, data.y)
    res["本文算法(轻量化PSO)"] = (mf.score(), time.perf_counter() - t0, mf.labels_)

    K = int(data.y.max()) + 1
    t0 = time.perf_counter()
    e3 = EntropyKPrototypes(n_clusters=K, seed=seed).fit_predict(data.X_num, data.X_cat, data.y)
    res["文献[3]熵聚类"] = (all_metrics(data.y, e3), time.perf_counter() - t0, e3)

    t0 = time.perf_counter()
    e4 = SMAFuzzyCMeans(n_clusters=K, seed=seed).fit_predict(data.X_num, data.X_cat)
    res["文献[4]黏菌FCM"] = (all_metrics(data.y, e4), time.perf_counter() - t0, e4)

    t0 = time.perf_counter()
    from knn_hybrid.baselines import wavelet_knn_baseline
    yp, acc, _ = wavelet_knn_baseline(data.X_num, data.X_cat, data.y, k=k)
    res["小波+kNN(消融)"] = (all_metrics(data.y, yp), time.perf_counter() - t0, yp)
    return res


def exp2_accuracy_vs_dim(datasets, dims=(4, 8, 12, 16, 20), seed=0):
    rows = []
    for d in datasets:
        for dim in dims:
            sub = take_dim(d, dim)
            noisy = add_noise(sub, 6.0, seed=seed + 7)
            res = _run_all(noisy, seed=seed)
            for algo, (met, t, _) in res.items():
                rows.append(dict(数据集=d.name, 属性维度=dim, 算法=algo,
                                 ACC=round(met["ACC"], 4), NMI=round(met["NMI"], 4),
                                 ARI=round(met["ARI"], 4), 时间s=round(t, 3)))
            print(f"  {d.name} dim={dim} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "accuracy_vs_dim.csv"), index=False, encoding="utf-8-sig")
    return df


# 聚类时间 vs 样本数
def exp3_time_vs_n(datasets, sizes=(100, 200, 300, 400), seed=0):
    rows = []
    rng = np.random.default_rng(seed)
    for d in datasets:
        base = add_noise(d, 6.0, seed=seed + 7)
        for n in sizes:
            idx = rng.choice(base.n, min(n, base.n), replace=False)
            sub = base.subset(idx)
            res = _run_all(sub, seed=seed)
            for algo, (met, t, _) in res.items():
                rows.append(dict(数据集=d.name, 样本数=n, 算法=algo,
                                 聚类时间s=round(t, 3), ACC=round(met["ACC"], 4)))
            print(f"  {d.name} n={n} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "time_vs_n.csv"), index=False, encoding="utf-8-sig")
    return df


def make_figures(df_den, df_acc, df_time):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # SNR
    piv = df_den.pivot_table(index=["数据类型", "数据集"], columns="方法", values="SNR_dB")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, tag in zip(axes, ["表格式 DS1", "平滑采样 DS1"]):
        sub = piv.loc[tag] if tag in piv.index.get_level_values(0) else None
        if sub is None:
            continue
        sub.T.plot(kind="bar", ax=ax, rot=20)
        ax.set_title(f"Denoising SNR ({tag})")
        ax.set_ylabel("SNR (dB)")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_denoise_snr.png"), dpi=130)
    plt.close(fig)

    # 准确率 vs 维度
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, name in zip(axes, df_acc["数据集"].unique()):
        sub = df_acc[df_acc["数据集"] == name]
        for algo, g in sub.groupby("算法"):
            g = g.sort_values("属性维度")
            ax.plot(g["属性维度"], g["ACC"], marker="o", label=algo)
        ax.set_title(name)
        ax.set_xlabel("num attributes")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("ACC")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_accuracy_vs_dim.png"), dpi=130)
    plt.close(fig)

    # 时间 vs 样本数
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, name in zip(axes, df_time["数据集"].unique()):
        sub = df_time[df_time["数据集"] == name]
        for algo, g in sub.groupby("算法"):
            g = g.sort_values("样本数")
            ax.plot(g["样本数"], g["聚类时间s"], marker="s", label=algo)
        ax.set_title(name)
        ax.set_xlabel("num samples")
    axes[0].set_ylabel("time (s)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_time_vs_n.png"), dpi=130)
    plt.close(fig)


def _show(df, title):
    print(f"\n=== {title} ===")
    try:
        print(df.to_string())
    except Exception:
        print(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["all", "1", "2", "3"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    ds = build_datasets(a.seed)
    print("datasets:", [(d.name, d.d_num, d.d_cat, int(d.y.max()) + 1) for d in ds], flush=True)

    df_den = df_acc = df_time = None
    if a.exp in ("all", "1"):
        df_den = exp1_denoise(a.seed)
        _show(df_den.pivot_table(index=["数据类型", "数据集"], columns="方法",
                                 values="SNR_dB").round(2), "实验1 去噪 SNR (dB)")
    if a.exp in ("all", "2"):
        dims = (4, 8, 16) if a.quick else (4, 8, 12, 16, 20)
        df_acc = exp2_accuracy_vs_dim(ds, dims=dims, seed=a.seed)
        _show(df_acc.pivot_table(index=["数据集", "属性维度"], columns="算法",
                                 values="ACC").round(3), "实验2 聚类准确率 ACC")
    if a.exp in ("all", "3"):
        sizes = (100, 400) if a.quick else (100, 200, 300, 400)
        df_time = exp3_time_vs_n(ds, sizes=sizes, seed=a.seed)
        _show(df_time.pivot_table(index=["数据集", "样本数"], columns="算法",
                                  values="聚类时间s").round(3), "实验3 聚类时间 (s)")
    if df_den is not None and df_acc is not None and df_time is not None:
        make_figures(df_den, df_acc, df_time)
        print("\nfigures saved to", RES)


if __name__ == "__main__":
    main()
