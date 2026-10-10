# -*- coding: utf-8 -*-
"""从论文图 2（SNR—运行时间）反推两条曲线的数据。"""
import numpy as np
from PIL import Image

a = np.asarray(Image.open("pdf_figs/p4_img1.png").convert("RGB")).astype(int)
r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]

X0, X1 = 84, 552
Y0, Y1 = 42, 450

notwhite = (r + g + b) < 735
green = notwhite & (g >= r - 10) & (g > b + 6)          # 所提算法：绿色面积
bluegray = notwhite & (b > r + 8) & (b >= g - 4) & (b < 245)  # 小波：蓝灰线


def y2snr(y):
    return 50.0 - (y - Y0) * (5.0 / 81.0)


def x2t(x):
    return 20.0 + (x - 149.0) * (10.0 / 65.5)


def curve_top(mask):
    out = {}
    for x in range(X0, X1):
        ys = np.where(mask[Y0:Y1, x])[0] + Y0
        if len(ys):
            out[x] = int(ys.min())
    return out


def curve_points(mask):
    out = {}
    for x in range(X0, X1):
        ys = np.where(mask[Y0:Y1, x])[0] + Y0
        if len(ys):
            # 取最长连续段的中点，抑制虚线网格干扰
            runs, s, prev = [], None, None
            for y in ys:
                if s is None:
                    s = prev = y
                elif y - prev <= 2:
                    prev = y
                else:
                    runs.append((s, prev))
                    s = prev = y
            if s is not None:
                runs.append((s, prev))
            best = max(runs, key=lambda t: t[1] - t[0])
            out[x] = (best[0] + best[1]) / 2.0
    return out


ours = curve_top(green)
wave = curve_points(bluegray)
print("green cols:", len(ours), " bluegray cols:", len(wave))

ts = np.arange(10, 91, 1.0)


def interp(pts):
    xs = np.array(sorted(pts))
    ys = np.array([pts[x] for x in xs])
    tt = x2t(xs)
    order = np.argsort(tt)
    return np.interp(ts, tt[order], ys[order])


o = interp(ours)
w = interp(wave)
snr_o = y2snr(o)
snr_w = y2snr(w)

print("\n t(s)  所提算法SNR  小波变换SNR")
for i in range(0, len(ts), 4):
    print(f" {ts[i]:5.1f}  {snr_o[i]:8.2f}  {snr_w[i]:11.2f}")

np.savetxt("fig2_snr_data.csv",
           np.column_stack([ts, snr_o, snr_w]),
           delimiter=",", header="time_s,snr_proposed_dB,snr_wavelet_dB", comments="", fmt="%.3f")
print("\nsaved fig2_snr_data.csv")
print("mean SNR  所提=%.2f dB  小波=%.2f dB" % (snr_o.mean(), snr_w.mean()))
