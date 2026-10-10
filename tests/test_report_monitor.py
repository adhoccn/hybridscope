# -*- coding: utf-8 -*-
"""v2.2 自检：数据指纹、HTML 报告、跨批次漂移判定。

运行：
  <py> tests/test_report_monitor.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybridscope.estimator import HybridScopeEstimator          # noqa: E402
from hybridscope.io import data_fingerprint, fingerprint_equal   # noqa: E402
from hybridscope import monitor, report                          # noqa: E402

N_PASS = 0
N_FAIL = 0


def check(name: str, ok: bool, detail: str = ""):
    global N_PASS, N_FAIL
    if ok:
        N_PASS += 1
        print("  [ok]   %s %s" % (name, ("— " + detail) if detail else ""))
    else:
        N_FAIL += 1
        print("  [FAIL] %s %s" % (name, ("— " + detail) if detail else ""))


def synth(n=180, m=4, n_clust=3, sep=1.6, seed=0):
    r = np.random.default_rng(seed)
    c = r.normal(0, sep, size=(n_clust, m))
    lab = r.integers(0, n_clust, n)
    return c[lab] + r.normal(0, 1.0, size=(n, m)), lab


print("== 1. 数据指纹 ==")
df = pd.DataFrame({
    "a": np.arange(30.0),
    "b": np.random.default_rng(1).normal(0, 1, 30),
    "班": ["甲", "乙"] * 14 + [None, "丙"],        # 类别列里混 NaN
})
fp1 = data_fingerprint(df)
check("类别列含 NaN 也能出指纹", fp1.get("hash_rows") is not None,
      str(fp1.get("hash_rows")))
shuf = df.sample(frac=1.0, random_state=2).reset_index(drop=True)
fp2 = data_fingerprint(shuf)
check("换顺序：hash_rows 变、hash_pool 不变",
      (not fingerprint_equal(fp1, fp2)) and fingerprint_equal(fp1, fp2, pool=True),
      "%s/%s vs %s/%s" % (fp1["hash_rows"][:8], fp1["hash_pool"][:8],
                          fp2["hash_rows"][:8], fp2["hash_pool"][:8]))
edit = df.copy()
edit.loc[0, "b"] = edit.loc[0, "b"] + 9.0
fp3 = data_fingerprint(edit)
check("改一格：两个哈希都变",
      (not fingerprint_equal(fp1, fp3)) and (not fingerprint_equal(fp1, fp3, pool=True)),
      fp3["hash_pool"][:8])
check("数值摘要齐全",
      fp1["numeric"]["b"]["mean"] is not None and fp1["numeric"]["b"]["q50"] is not None,
      "mean=%.3f q50=%.3f" % (fp1["numeric"]["b"]["mean"], fp1["numeric"]["b"]["q50"]))

print("\n== 2. HTML 报告 ==")
X, y = synth(n=150, m=4, seed=5)
dfx = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
est = HybridScopeEstimator(k=5, run_stability=False, n_boot=0, seed=0).fit(dfx, y)
html = report.build_html(est, df=dfx, source="synth", title="自检报告")
for sec in ("零、复现信息", "四、特征权重与可信度", "六、解读与下一步"):
    check("含章节 %s" % sec, sec in html)
check("含数据指纹", (data_fingerprint(dfx)["hash_rows"] in html))
check("含内嵌图", html.count("<img src='data:image/png;base64,") >= 1,
      "%d 张" % html.count("<img src='data:image/png;base64,"))
acts = report.action_list(est)
check("动作清单非空且带档位", len(acts) > 0 and all(lv in ("green", "amber", "red", "grey")
                                               for lv, _ in acts),
      "%d 条：%s" % (len(acts), acts[0][1][:40]))
tmp = os.path.join(tempfile.gettempdir(), "_hs_report.html")
report.write_report(tmp, est, df=dfx)
check("可写盘且非空", os.path.getsize(tmp) > 2000, "%d 字节" % os.path.getsize(tmp))

print("\n== 3. 无标签路径不炸 ==")
est_u = HybridScopeEstimator(k=5, run_stability=False, n_boot=0, seed=0,
                             pso_particles=10, pso_iters=10).fit(dfx, None)
html_u = report.build_html(est_u, df=dfx)
check("无标签也能出报告", "六、解读与下一步" in html_u)
check("无标签权重非空", est_u.w_ is not None,
      np.round(est_u.w_, 3).tolist().__str__())

print("\n== 4. 漂移判定：同一批数据、换种子 ==")
Xs, ys = synth(n=200, m=4, sep=1.6, seed=11)
dfs = pd.DataFrame(Xs, columns=["f1", "f2", "f3", "f4"])
opt = dict(k=5, run_stability=False, n_boot=0, pso_particles=10, pso_iters=10)
e1 = HybridScopeEstimator(seed=1, **opt).fit(dfs, ys)
e2 = HybridScopeEstimator(seed=77, **opt).fit(dfs, ys)
s1 = e1.snapshot(df=dfs, name="A")
s2 = e2.snapshot(df=dfs, name="B")
r_same = monitor.compare(s1, s2)
check("数据指纹一致被认出", r_same["data"]["same_data_pool"])
check("判定为「同一批数据」，不是漂移", r_same["kind"] == "same_data",
      "%s | L1=%.4f" % (r_same["kind"], r_same["l1"]))
check("有可用标准误（否则这项没意义）", r_same["n_testable"] > 0,
      "%d 个可检验" % r_same["n_testable"])

print("\n== 5. 漂移判定：真漂移 ==")
Xb, yb = synth(n=400, m=4, sep=1.6, seed=11)          # 与 A 同规模，只改一处结构
dfb = pd.DataFrame(Xb, columns=["f1", "f2", "f3", "f4"])
eb = HybridScopeEstimator(seed=1, **opt).fit(dfb, yb)
sb = eb.snapshot(df=dfb, name="B0")
Xd, yd = synth(n=400, m=4, sep=1.6, seed=23)
Xd[:, 0] += 3.0 * (yd == 0)          # 让 f1 在新批次里变得更能区分簇
dfd = pd.DataFrame(Xd, columns=["f1", "f2", "f3", "f4"])
e3 = HybridScopeEstimator(seed=1, **opt).fit(dfd, yd)
s3 = e3.snapshot(df=dfd, name="C")
r_drift = monitor.compare(sb, s3)
check("指纹识别为不同数据", not r_drift["data"]["same_data_pool"])
check("数据侧检出 f1 漂移", "f1" in r_drift["data"]["flagged"],
      "flagged=%s" % r_drift["data"]["flagged"])
check("权重侧给出判定", r_drift["n_testable"] > 0 and len(r_drift["rows"]) > 0,
      "可检验 %d，显著 %d" % (r_drift["n_testable"], r_drift["n_sig"]))
check("整体或逐项至少一处判为漂移",
      bool(r_drift["n_sig"] > 0 or (r_drift.get("p_global") or 1.0) < 0.05),
      "chi2=%.2f p=%s n_sig=%d" % (r_drift["chi2"], r_drift.get("p_global"),
                                   r_drift["n_sig"]))
txt = monitor.to_text(r_drift)
check("文本结论可读", "结论：" in txt and "权重侧" in txt)

print("\n== 6. 缺标准误时如实说不可判 ==")
u1 = HybridScopeEstimator(seed=1, **opt).fit(dfs, None).snapshot(df=dfs, name="U1")
u2 = HybridScopeEstimator(seed=2, **opt).fit(dfd, None).snapshot(df=dfd, name="U2")
r_no = monitor.compare(u1, u2)
check("无标签两份 → 判为缺标准误", r_no["kind"] == "no_se",
      "%s（可检验 %d）" % (r_no["kind"], r_no["n_testable"]))
check("逐项标记为不可判", all(r["verdict"] == "不可判" for r in r_no["rows"]))

print("\n== 7. 快照存档与回读 ==")
p = os.path.join(tempfile.gettempdir(), "_hs_snap.json")
monitor.save_snapshot(s1, p)
back = monitor.load_snapshot(p)
check("回读后权重一致", np.allclose(back["w"], s1["w"]))
check("回读后指纹一致", back["fingerprint"]["hash_rows"] == s1["fingerprint"]["hash_rows"])
r_round = monitor.compare(back, s1)
check("回读后与自身对比判为同数据", r_round["kind"] == "same_data")

print("\n== 8. 同名不同义的列不参与权重对比 ==")
def snap_of(frame, w, se, names):
    return {"name": "x", "fingerprint": data_fingerprint(frame),
            "feature_names": names, "w": w, "se_w": se, "n": len(frame)}

f1 = pd.DataFrame({"班": ["A", "B", "C"] * 20, "v": np.arange(60.0)})
f2 = pd.DataFrame({"班": ["X", "Y", "Z"] * 20, "v": np.arange(60.0) + 0.5})
# 共同特征只留「班」这一列：它的取值域完全不重叠，于是无处可比
r_ren = monitor.compare(snap_of(f1, [1.0], [0.05], ["班"]),
                        snap_of(f2, [1.0], [0.05], ["班"]))
check("取值域不重叠被识别", "班" in r_ren["data"]["renamed"],
      "renamed=%s flagged=%s" % (r_ren["data"]["renamed"], r_ren["data"]["flagged"]))
check("该列判为不可比", r_ren["rows"][0]["verdict"] in ("不可比", "口径不可比"),
      r_ren["rows"][0]["why"])
check("整体判为无法比较", r_ren["kind"] == "not_comparable", r_ren["kind"])

print("\n%d 项通过，%d 项失败" % (N_PASS, N_FAIL))
sys.exit(1 if N_FAIL else 0)
