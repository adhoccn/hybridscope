# -*- coding: utf-8 -*-
"""OCR 论文位图，输出文字及其位置（用于读取坐标轴刻度与图例）。"""
import sys

from rapidocr_onnxruntime import RapidOCR

scale = 3.0
eng = RapidOCR()
for path in sys.argv[1:]:
    res, _ = eng(path)
    print(f"=== {path}")
    if not res:
        print("  (none)")
        continue
    for box, text, score in res:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, x1 = min(xs) / scale, max(xs) / scale
        y0, y1 = min(ys) / scale, max(ys) / scale
        print(f"  '{text}'  score={score}  x[{x0:.0f},{x1:.0f}] y[{y0:.0f},{y1:.0f}]")
