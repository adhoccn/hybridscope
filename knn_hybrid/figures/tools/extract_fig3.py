# -*- coding: utf-8 -*-
'''从论文图 3（三张子图）反推"算法 × 数据属性数"的聚类性能（归一化相对值）。'''
import numpy as np
from PIL import Image

CONFIG = {
    "pdf_figs/p5_img0.png": dict(up=(243, 635, 58, 292), dn=(379, 612), lx=(86, 320), rx=(395, 633)),
    "pdf_figs/p5_img1.png": dict(up=(244, 648, 74, 314), dn=(404, 647), lx=(81, 323), rx=(402, 646)),
    "pdf_figs/p5_img2.png": dict(up=(238, 638, 73, 311), dn=(400, 641), lx=(77, 316), rx=(394, 636)),
}
DS = {"pdf_figs/p5_img0.png": "DS1", "pdf_figs/p5_img1.png": "DS2", "pdf_figs/p5_img2.png": "DS3"}

rows = []
for f, cfg in CONFIG.items():
    a = np.asarray(Image.open(f).convert("RGB")).astype(int)
    s = a.sum(axis=2)
    data = (s < 745) & (s > 150)

    def vertical_groups(x0, x1, y0, y1, g=4):
        sub = data[y0 + 2:y1 - 1, x0:x1]
        cols = np.where(sub.any(axis=0))[0]
        if len(cols) == 0:
            return [0.0] * g
        xa, xb = cols.min(), cols.max()
        edges = np.linspace(xa, xb + 1, g + 1).astype(int)
        vals = []
        for i in range(g):
            blk = sub[:, edges[i]:max(edges[i + 1], edges[i] + 1)]
            ys = np.where(blk.any(axis=1))[0]
            vals.append((y1 - y0 - float(np.quantile(ys, 0.10))) / (y1 - y0) if len(ys) else 0.0)
        return vals

    def horizontal_groups(x0, x1, y0, y1, g=4):
        sub = data[y0 + 2:y1 - 1, x0:x1]
        rws = np.where(sub.any(axis=1))[0]
        if len(rws) == 0:
            return [0.0] * g
        ya, yb = rws.min(), rws.max()
        edges = np.linspace(ya, yb + 1, g + 1).astype(int)
        vals = []
        for i in range(g):
            blk = sub[edges[i]:max(edges[i + 1], edges[i] + 1), :]
            xs = np.where(blk.any(axis=0))[0]
            vals.append(float(np.quantile(xs, 0.92)) / (x1 - x0) if len(xs) else 0.0)
        return vals

    ux0, ux1, uy0, uy1 = cfg["up"]
    dy0, dy1 = cfg["dn"]
    lx0, lx1 = cfg["lx"]
    rx0, rx1 = cfg["rx"]
    ours = vertical_groups(ux0, ux1, uy0, uy1)
    ref3 = horizontal_groups(lx0, lx1, dy0, dy1)
    ref4 = horizontal_groups(rx0, rx1, dy0, dy1)
    print(f"\n=== {DS[f]} ({f})")
    print("  所提算法  :", np.round(ours, 3))
    print("  文献[3]算法:", np.round(ref3, 3))
    print("  文献[4]算法:", np.round(ref4, 3))
    for gi in range(4):
        rows.append((DS[f], gi + 1, ours[gi], ref3[gi], ref4[gi]))

with open("fig3_raw_relative.csv", "w", encoding="utf-8") as fh:
    fh.write("dataset,attr_num,proposed_rel,ref3_rel,ref4_rel\n")
    for r in rows:
        fh.write("%s,%d,%.4f,%.4f,%.4f\n" % r)
print("\nsaved fig3_raw_relative.csv")
