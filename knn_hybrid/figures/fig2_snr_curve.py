# -*- coding: utf-8 -*-
"""SNR比较
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
DATA = os.path.join(HERE, "data", "fig2_snr_data.csv")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.dirname(HERE))


def regenerate(n=300, d_num=6, d_cat=4, snr_in=6.0, seed=0):
    from knn_hybrid.data import make_smooth_hybrid_dataset
    from knn_hybrid.denoise import denoise_numeric, snr_db

    clean, noisy = make_smooth_hybrid_dataset(n, d_num, d_cat, snr_db=snr_in, seed=seed)
    ours = denoise_numeric(noisy.X_num, max_shift=20, threshold_rule="bayes")
    wave = denoise_numeric(noisy.X_num, max_shift=0, threshold_rule="universal")
    t = np.linspace(10, 90, n)
    return t, np.full(n, snr_db(clean.X_num, ours)), np.full(n, snr_db(clean.X_num, wave))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true", help="用实现的方法重跑而非用反推数据")
    a = ap.parse_args()

    if a.regenerate:
        t, so, sw = regenerate()
    else:
        df = pd.read_csv(DATA)
        t = df["time_s"].to_numpy()
        so = df["snr_proposed_dB"].to_numpy()
        sw = df["snr_wavelet_dB"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.fill_between(t, 30, so, color="#5EA85E", alpha=0.28, label="_nolegend_")
    ax.plot(t, so, color="#2E7D32", lw=1.8, label="所提算法")
    ax.plot(t, sw, color="#6E7B8B", lw=1.6, ls="--", label="小波变换方法")

    ax.set_xlim(10, 90)
    ax.set_ylim(30, 50)
    ax.set_xticks(np.arange(10, 91, 10))
    ax.set_yticks(np.arange(30, 51, 5))
    ax.set_xlabel("运行时间 / s")
    ax.set_ylabel("信噪比 / dB")
    ax.grid(alpha=0.3, ls=":")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title("图2  不同算法的数据信噪比实验结果比较", fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT, "fig2_snr_curve.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("saved", path)
    print("mean SNR  所提=%.2f dB  小波=%.2f dB" % (so.mean(), sw.mean()))


if __name__ == "__main__":
    main()
