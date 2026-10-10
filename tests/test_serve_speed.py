# -*- coding: utf-8 -*-
"""v2.3 自检：算力预算、无损加速、HTTP 服务层。

运行：
  <py> tests/test_serve_speed.py
"""

from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybridscope import speed                                  # noqa: E402
from hybridscope.estimator import HybridScopeEstimator          # noqa: E402
from hybridscope.io import split_columns                        # noqa: E402
from knn_hybrid.model import WeightedDistance, knn_indices      # noqa: E402

N_PASS = 0
N_FAIL = 0


def check(name: str, ok: bool, detail: str = ""):
    global N_PASS, N_FAIL
    if ok:
        N_PASS += 1
        print("  PASS  %s" % name)
    else:
        N_FAIL += 1
        print("  FAIL  %s  %s" % (name, detail))


rng = np.random.default_rng(11)

print("== 1. 无损加速：缓存与低内存必须与原来等价 ==")
X = rng.normal(0, 1, (120, 4))
C = rng.integers(0, 3, (120, 2))
w = np.array([0.3, 0.2, 0.1, 0.1, 0.2, 0.1])
c = 1.0 + 0.3 * rng.random(120)
d_eager = WeightedDistance(X, C, low_memory=False)(w, c)
d_lazy = WeightedDistance(X, C, low_memory=True)(w, c)
check("低内存模式数值等价", np.allclose(d_eager, d_lazy, atol=0, rtol=0)
      or np.abs(d_eager - d_lazy).max() < 1e-12,
      "最大差 %.3g" % np.abs(d_eager - d_lazy).max())


def _knn_ref(D, k, exclude_self=True, ordered=True):
    n = D.shape[0]
    k = int(max(1, min(k, n - 1)))
    Dm = D.copy()
    if exclude_self:
        np.fill_diagonal(Dm, np.inf)
    idx = np.argpartition(Dm, k - 1, axis=1)[:, :k]
    if not ordered:
        return idx
    rows = np.arange(n)[:, None]
    return idx[rows, np.argsort(Dm[rows, idx], axis=1)]


for ordered in (True, False):
    ok = np.array_equal(knn_indices(d_eager, 5, True, ordered),
                        _knn_ref(d_eager, 5, True, ordered))
    check("近邻索引与参考实现一致（ordered=%s）" % ordered, ok)
check("近邻不含自身", not (knn_indices(d_eager, 5) == np.arange(120)[:, None]).any())

print("\n== 2. 算力预算：规模越大档位越低，且可审计 ==")
b_small = speed.budget(150, 6, 0, "auto")
b_big = speed.budget(4000, 8, 1, "auto")
check("小表保持默认预算", b_small["pso_particles"] == 15 and b_small["pso_iters"] == 20,
      str(b_small))
check("大表降低 PSO 预算",
      (b_big["pso_particles"] * b_big["pso_iters"])
      < (b_small["pso_particles"] * b_small["pso_iters"]), str(b_big))
check("大表启用子采样", b_big["subsample_n"] is not None, str(b_big["subsample_n"]))
check("大表启用低内存", b_big["low_memory"] is True)
b_fast = speed.budget(4000, 8, 1, "fast")
check("fast 比 auto 更省", (b_fast["pso_particles"] * b_fast["pso_iters"])
      <= (b_big["pso_particles"] * b_big["pso_iters"]))
b_th = speed.budget(4000, 8, 1, "thorough")
check("thorough 不压缩", b_th["pso_particles"] == 15 and b_th["pso_iters"] == 20)

est_cost = speed.cost_estimate(400, 6, 0)
check("耗时预估随规模单调", speed.cost_estimate(800, 6, 0)["seconds_total"]
      > est_cost["seconds_total"] * 2,
      "%.2f vs %.2f" % (speed.cost_estimate(800, 6, 0)["seconds_total"],
                        est_cost["seconds_total"]))
check("内存预估随 n² 增长",
      speed.dist_matrix_bytes(2000) >= 4 * speed.dist_matrix_bytes(1000) * 0.999,
      "%.0f vs %.0f" % (speed.dist_matrix_bytes(2000),
                        4 * speed.dist_matrix_bytes(1000)))
check("低内存省掉类别张量",
      speed.dist_matrix_bytes(1000, 5, low_memory=True)
      < speed.dist_matrix_bytes(1000, 5, low_memory=False))
txt = speed.advice(5000, 8, 1)
check("预估文本给出建议", "建议" in txt and "分钟" in txt or "min" in txt, txt[:80])

print("\n== 3. 预算真的被用上，且显式参数不被覆盖 ==")
n = 160
df = pd.DataFrame(rng.normal(0, 1, (n, 5)), columns=[f"x{i}" for i in range(5)])
df["y"] = (df["x0"] * 0.8 + rng.normal(0, 0.4, n) > 0).astype(int)
Xn, Xc, nn, cn, yy = split_columns(df, target="y", cat_cols=[])
e1 = HybridScopeEstimator(k=5, seed=0, auto_prep=False, speed="fast",
                          run_stability=False, n_boot=0)
e1.fit_hybrid(Xn, Xc, yy, feature_names=list(nn))
check("fast 模式记录预算", (e1.speed_ or {}).get("mode") == "fast", str(e1.speed_))
check("fast 模式实际用了更少的粒子", e1.pso_particles == 8, str(e1.pso_particles))
e2 = HybridScopeEstimator(k=5, seed=0, auto_prep=False, speed="auto",
                          run_stability=False, n_boot=7,
                          speed_keep=("n_boot",))
e2.fit_hybrid(Xn, Xc, yy, feature_names=list(nn))
check("显式给出的参数不被预算覆盖", e2.n_boot == 7, str(e2.n_boot))

print("\n== 4. HTTP 服务层 ==")
try:
    from fastapi.testclient import TestClient
    from hybridscope.serve import app
except Exception as exc:                      # pragma: no cover
    print("  SKIP  服务层依赖缺失：%s" % exc)
    TestClient = None

if TestClient is not None:
    client = TestClient(app)
    r = client.get("/health")
    check("健康检查", r.status_code == 200 and r.json()["ok"] is True)

    n2 = 120
    d2 = pd.DataFrame(rng.normal(0, 1, (n2, 4)), columns=list("abcd"))
    score = d2["a"] * 0.7 + d2["b"] * 0.3
    d2["总评"] = np.round(60 + 12 * score, 1)
    d2["班"] = ["甲", "乙", "丙"] * (n2 // 3)
    csv_text = d2.to_csv(index=False)

    r = client.post("/analyze/text", json={"csv": csv_text, "target": "总评",
                                           "cat": ["班"], "speed": "fast",
                                           "boot": 10, "stab_boot": 10,
                                           "stab_perm": 10, "no_stability": True})
    check("分析接口返回 200", r.status_code == 200, r.text[:200])
    if r.status_code == 200:
        d = r.json()
        check("返回权重与诊断", bool(d.get("weights")) and bool(d.get("diagnostics")),
              str(list(d)[:8]))
        check("连续标签给出警告", any("连续分数" in w for w in d.get("warnings", [])),
              str(d.get("warnings")))
        check("返回可存档快照", "w" in (d.get("snapshot") or {}),
              str(list((d.get("snapshot") or {}))[:8]))
        try:
            json.dumps(d, allow_nan=False)          # NaN/Inf 会让这里抛异常
            no_nan = True
        except ValueError:
            no_nan = False
        check("输出不含 NaN（可 JSON 序列化）", no_nan)
        rep = client.post("/compare", json={"a": d["snapshot"], "b": d["snapshot"]})
        check("自身对比判为同数据", rep.json().get("kind") == "same_data",
              str(rep.json().get("kind")))

    r = client.post("/analyze/text", json={"csv": csv_text, "target": "总评",
                                           "cat": ["班"], "band": 3, "speed": "fast",
                                           "no_stability": True, "boot": 0})
    if r.status_code == 200:
        check("切档后原分数不留在特征里",
              "总评" not in (r.json().get("weights") or {}),
              str(list((r.json().get("weights") or {}))))
    else:
        check("切档后原分数不留在特征里", False, r.text[:200])

    r = client.post("/analyze/text", json={"csv": csv_text})
    check("没有标签列时返回 422", r.status_code == 422, str(r.status_code))

    r = client.post("/advice", files={"file": ("t.csv", csv_text, "text/csv")},
                    data={"target": "总评"})
    check("算力预估接口", r.status_code == 200 and "budgets" in r.json(),
          str(r.status_code))

print("\n%d 项通过，%d 项失败" % (N_PASS, N_FAIL))
sys.exit(1 if N_FAIL else 0)
