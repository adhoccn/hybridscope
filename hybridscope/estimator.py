# -*- coding: utf-8 -*-
"""sklearn 兼容估计器：fit / predict / diagnose 三件套。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from hybridscope.advisor import recommend
from hybridscope.diagnostics import diagnose
from hybridscope.io import data_fingerprint, split_columns
from hybridscope.prep import Prep
from hybridscope.stability import fit_weights_unsupervised, label_free_credibility
from knn_hybrid.model import WeightedDistance
from knn_hybrid.pipeline import KNNHybridClustering

TARGET = "__target__"


class HybridScopeEstimator(BaseEstimator, ClassifierMixin):
    """混合属性分型器。denoise='auto' 时由去噪顾问决定开关。"""

    def __init__(self, k: int = 5, denoise: str | bool = "auto", wavelet: str = "db4",
                 level: int = 3, pso_particles: int = 15, pso_iters: int = 20,
                 contribution_rounds: int = 2, seed: int = 0,
                 run_diagnose: bool = True, n_boot: int = 200,
                 auto_prep: bool = True, prep_kw: dict | None = None,
                 fdr_alpha: float = 0.05, run_stability: bool = True,
                 stab_boot: int = 60, stab_perm: int = 60,
                 n_boot_lofo: int = 15, stab_max_n: int = 3000,
                 speed: str = "off", max_n: int | None = None,
                 speed_keep: tuple = ()):
        self.k = k
        self.denoise = denoise
        self.wavelet = wavelet
        self.level = level
        self.pso_particles = pso_particles
        self.pso_iters = pso_iters
        self.contribution_rounds = contribution_rounds
        self.seed = seed
        self.run_diagnose = run_diagnose
        self.n_boot = n_boot
        self.auto_prep = auto_prep
        self.prep_kw = prep_kw
        self.fdr_alpha = fdr_alpha
        self.run_stability = run_stability
        self.stab_boot = stab_boot
        self.stab_perm = stab_perm
        self.n_boot_lofo = n_boot_lofo
        self.stab_max_n = stab_max_n
        # speed='off' 时完全保留调用方给的参数（默认，行为与 2.2 一致）；
        # 设为 auto/fast/thorough 后按规模覆盖预算，实际用量存于 self.speed_。
        self.speed = speed
        self.max_n = max_n
        # speed_keep：调用方在命令行显式给过的参数，预算不覆盖它们
        self.speed_keep = tuple(speed_keep or ())

    def fit(self, X, y=None):
        """X 为 DataFrame（自动分列）或 ndarray（全数值）。

        auto_prep=True（默认）时先跑数据体检与修补：缺失填补、离群截断、
        低频档合并、标识列剔除、量纲失衡时标准化，处理过程存于 self.prep_。
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            if y is not None:
                df[TARGET] = np.asarray(y)
            # 指纹按「送进来的原始数据」算，用于跨批次核对是不是同一批人
            try:
                self.data_fingerprint_ = data_fingerprint(
                    df, target=TARGET if y is not None else None)
            except Exception:
                self.data_fingerprint_ = None
            if self.auto_prep:
                self.prep_ = Prep(**(self.prep_kw or {}))
                df = self.prep_.fit_transform(
                    df, target=TARGET if y is not None else None)
            else:
                self.prep_ = None
            X_num, X_cat, num_names, cat_names, yy = split_columns(
                df, target=TARGET if y is not None else None)
            self.num_names_ = num_names
            self.cat_names_ = cat_names
            self.cat_levels_ = [list(pd.unique(X[c])) for c in cat_names]
        else:
            X_num = np.asarray(X, dtype=float)
            X_cat = np.zeros((X_num.shape[0], 0), dtype=int)
            self.num_names_ = [f"x{i}" for i in range(X_num.shape[1])]
            self.cat_names_ = []
            self.cat_levels_ = []
            self.data_fingerprint_ = None
            yy = None if y is None else np.asarray(
                pd.Categorical(y).codes
                if not np.issubdtype(np.asarray(y).dtype, np.number) else y, dtype=int)
        return self.fit_hybrid(X_num, X_cat, yy)

    def fit_hybrid(self, X_num, X_cat=None, y=None, feature_names=None):
        X_num = np.asarray(X_num, float)
        X_cat = (np.asarray(X_cat, int) if X_cat is not None
                 else np.zeros((X_num.shape[0], 0), int))
        yy = None if y is None else np.asarray(y, int)
        if feature_names is not None:
            self.feature_names_ = list(feature_names)
            self.num_names_ = list(feature_names)
            self.cat_names_ = []
            self.cat_levels_ = []
        else:
            if not hasattr(self, "num_names_"):
                self.num_names_ = [f"x{i}" for i in range(X_num.shape[1])]
                self.cat_names_ = [f"c{i}" for i in range(X_cat.shape[1])]
                self.cat_levels_ = []
            self.feature_names_ = list(self.num_names_) + list(self.cat_names_)

        self.y_ = yy
        n = X_num.shape[0]
        self.k_ = int(min(self.k, max(n - 1, 1)))

        # ---- 算力预算：先按规模定档，再按档跑（见 hybridscope.speed）----
        self.speed_ = None
        low_memory = False
        if str(self.speed) in ("auto", "fast", "thorough"):
            from hybridscope.speed import budget as _budget
            b = _budget(n, X_num.shape[1],
                        X_cat.shape[1] if X_cat is not None else 0,
                        mode=str(self.speed), run_stability=self.run_stability)
            b["n_total"] = n
            self.speed_ = b
            keep = set(self.speed_keep)
            if "pso_particles" not in keep:
                self.pso_particles = b["pso_particles"]
            if "pso_iters" not in keep:
                self.pso_iters = b["pso_iters"]
            if "contribution_rounds" not in keep:
                self.contribution_rounds = b["contribution_rounds"]
            if "stab_boot" not in keep:
                self.stab_boot = b["stab_boot"]
            if "stab_perm" not in keep:
                self.stab_perm = b["stab_perm"]
            if "n_boot" not in keep:
                self.n_boot = b["n_boot"]
            low_memory = bool(b.get("low_memory"))
            # speed_ 记录的是「实际用了多少」，不是「预算推荐了多少」：
            # 调用方显式给过的参数要保留，读报告的人也得看到真值。
            for _k in ("pso_particles", "pso_iters", "contribution_rounds",
                       "stab_boot", "stab_perm", "n_boot"):
                b[_k] = getattr(self, _k)
            cap = self.max_n if self.max_n else b.get("subsample_n")
            if cap:
                b["subsample_n"] = min(int(cap), n)
        else:
            low_memory = False

        if self.denoise == "auto":
            self.advisor_ = recommend(X_num, X_cat, yy, k=self.k_,
                                      wavelet=self.wavelet, level=self.level,
                                      seed=self.seed)
            use_denoise = bool(self.advisor_["denoise"])
        else:
            self.advisor_ = None
            use_denoise = bool(self.denoise)

        self.model_ = KNNHybridClustering(
            k=self.k_, denoise=use_denoise, wavelet=self.wavelet, level=self.level,
            pso_particles=self.pso_particles, pso_iters=self.pso_iters,
            contribution_rounds=self.contribution_rounds, seed=self.seed,
            low_memory=low_memory)
        self.model_.fit(X_num, X_cat, yy)

        self.X_num_ = self.model_.X_num_
        self.X_cat_ = self.model_.X_cat_
        self.labels_ = self.model_.labels_
        self.w_ = self.model_.w_
        self.metrics_ = self.model_.score(yy) if yy is not None else None

        # 无标签：等权聚完就结束是原实现的短板。这里补一条不依赖 y 的路：
        # 以轮廓系数为目标拟合权重，再用 bootstrap 稳定性 + 置换零分布给可信度。
        self.unsupervised_ = None
        if yy is None and self.w_ is None and self.X_num_.shape[1] > 0:
            try:
                self.unsupervised_ = fit_weights_unsupervised(
                    self.X_num_, self.X_cat_, k=self.k_,
                    particles=max(8, self.pso_particles - 3),
                    iters=max(8, self.pso_iters - 8), seed=self.seed)
                self.w_ = self.unsupervised_["w"]
                self.labels_ = self.unsupervised_["labels"]
                self.metrics_ = {"silhouette": self.unsupervised_["silhouette"],
                                 "n_clusters": self.unsupervised_["n_clusters"]}
            except Exception as exc:
                self.unsupervised_ = {"error": str(exc)}

        self.diagnostics_ = None
        self.stability_ = None

        if yy is not None and self.run_diagnose and self.w_ is not None:
            try:
                self.diagnostics_ = diagnose(
                    self.X_num_, self.X_cat_, yy, self.w_, k=self.k_,
                    feature_names=self.feature_names_, fdr_alpha=self.fdr_alpha,
                    n_boot=self.n_boot, seed=self.seed)
                # 诊断会在 θ* 处做一次局部精修，精修后的点才是"带标准误的那一个"。
                # 权重与标准误必须来自同一个点，否则快照存的是 PSO 的输出、
                # 显著性却算在精修点上，跨批次 Δw 就混进了两次优化的差。
                w_ref = np.asarray(self.diagnostics_.get("w"), float)
                if (w_ref.size == np.asarray(self.w_, float).size
                        and np.all(np.isfinite(w_ref))
                        and (self.diagnostics_.get("identifiable") is not False)):
                    self.w_pso_ = np.asarray(self.w_, float)
                    self.w_ = w_ref
            except Exception as exc:
                self.diagnostics_ = {"error": str(exc)}

        # 稳定性检验不依赖标签，两种情形都跑：有标签时它是对 sandwich 的交叉验证，
        # 无标签时它就是唯一的可信度来源。
        if self.run_stability and self.w_ is not None and n <= int(self.stab_max_n):
            try:
                # 子采样：观测值与零分布必须用同一个子样本规模，否则两边方差
                # 不可比，z 值会失真。这里只切样本，不切零分布的次数。
                idx = None
                cap = (self.speed_ or {}).get("subsample_n")
                if cap and n > int(cap):
                    from hybridscope.speed import subsample_indices
                    idx = subsample_indices(n, cap, seed=self.seed)
                Xn = self.X_num_ if idx is None else self.X_num_[idx]
                Xc = self.X_cat_ if idx is None else self.X_cat_[idx]
                self.stability_ = label_free_credibility(
                    Xn, Xc, self.w_, k=self.k_,
                    n_boot=self.stab_boot, n_perm=self.stab_perm, seed=self.seed,
                    feature_names=self.feature_names_,
                    n_boot_lofo=self.n_boot_lofo)
                if idx is not None and isinstance(self.stability_, dict):
                    self.stability_["n_used"] = int(len(idx))
                    self.stability_["subsampled_from"] = int(n)
            except Exception as exc:
                self.stability_ = {"error": str(exc)}
        if self.speed_ is not None:
            self.speed_["seconds_pso"] = getattr(getattr(self, "model_", None),
                                                 "time_pso_", None)
        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_

    def predict(self, X, X_cat=None):
        """新数据走同一套修补规则（缺失/离群/低频档/量纲），再算邻居投票。"""
        if getattr(self, "prep_", None) is not None and isinstance(X, pd.DataFrame):
            X = self.prep_.transform(X)
        if X_cat is not None:
            X_num = np.asarray(X, dtype=float)
            X_cat = np.asarray(X_cat, int)
        else:
            X_num, X_cat, _ = self._encode(X)
        n_tr = self.X_num_.shape[0]
        wd = WeightedDistance(np.vstack([self.X_num_, X_num]),
                              np.vstack([self.X_cat_, X_cat]))
        c = np.concatenate([self.model_.c_ if self.model_.c_ is not None else np.ones(n_tr),
                            np.ones(X_num.shape[0])])
        D = wd(self.w_, c)
        Dsub = D[n_tr:, :n_tr]
        idx = np.argpartition(Dsub, min(self.k_, n_tr) - 1, axis=1)[:, : self.k_]
        n_cls = int(self.y_.max()) + 1
        cnt = np.stack([(self.y_[idx] == c_).sum(axis=1) for c_ in range(n_cls)], axis=1)
        return cnt.argmax(axis=1)

    def _encode(self, X):
        """按训练时的列定义与类别编码，把新数据转成矩阵。"""
        if isinstance(X, pd.DataFrame):
            num = [c for c in self.num_names_ if c in X.columns]
            cat = [c for c in self.cat_names_ if c in X.columns]
            X_num = X[num].to_numpy(float) if num else np.zeros((len(X), self.X_num_.shape[1]))
            cols = []
            for c in cat:
                levels = self.cat_levels_[self.cat_names_.index(c)]
                m = {v: i for i, v in enumerate(levels)}
                cols.append(X[c].map(m).fillna(0).to_numpy(int))
            X_cat = np.column_stack(cols) if cols else np.zeros((len(X), self.X_cat_.shape[1]), int)
        else:
            X_num = np.asarray(X, dtype=float)
            X_cat = np.zeros((X_num.shape[0], self.X_cat_.shape[1]), int)
        return X_num, X_cat, None

    def summary(self) -> dict:
        out = {
            "n": int(self.X_num_.shape[0]),
            "k": self.k_,
            "denoise": None if self.advisor_ is None else self.advisor_["denoise"],
            "metrics": self.metrics_,
        }
        if self.advisor_ is not None:
            out["advisor"] = {
                k: self.advisor_.get(k) for k in
                ("rule", "acf1_median", "tau_median", "tau_band", "sort_key",
                 "delta_acc", "caveat")
                if self.advisor_.get(k) is not None
            }
        if getattr(self, "prep_", None) is not None:
            out["prep"] = self.prep_.report()
        fp = getattr(self, "data_fingerprint_", None)
        if fp is not None:
            out["fingerprint"] = {"hash_rows": fp["hash_rows"],
                                  "hash_pool": fp["hash_pool"],
                                  "n_rows": fp["n_rows"], "n_cols": fp["n_cols"]}
        out["weights"] = (dict(zip(self.feature_names_,
                                   np.round(self.w_, 4).tolist()))
                          if self.w_ is not None else None)
        if getattr(self, "speed_", None):
            out["speed"] = dict(self.speed_)
            out["speed"]["pso_seconds"] = getattr(getattr(self, "model_", None),
                                                  "time_pso_", None)
        if self.diagnostics_ and "rows" in self.diagnostics_:
            d = self.diagnostics_
            out["diagnostics"] = d["rows"]
            out["fdr"] = {"alpha": d["fdr_alpha"], "n_tests": d["n_tests"],
                          "n_sig": d["n_sig"]}
            out["rho_trace"] = d["rho_trace"]
            out["identifiable"] = d["identifiable"]
            out["unstable"] = d["unstable"]
            out["unstable_reason"] = d["unstable_reason"]
            out["separable"] = d["separable"]
            out["weights_at_diagnosis"] = dict(
                zip(self.feature_names_, np.round(d["w"], 4).tolist()))
        if self.stability_ and "level" in self.stability_:
            s = self.stability_
            out["stability"] = {
                "level": s["level"], "level_zh": s["level_zh"],
                "silhouette": s["silhouette"], "stability": s["stability"],
                "stability_sd": s["stability_sd"], "null_mean": s["null_mean"],
                "null_sd": s["null_sd"], "z": s["z"], "p_null": s["p_null"],
                "gauss_mean": s.get("gauss_mean"), "z_gauss": s.get("z_gauss"),
                "p_gauss": s.get("p_gauss"), "basis": s.get("basis"),
                "n_clusters": s["n_clusters"], "verdict": s["verdict"],
                "top_features": s.get("top_features", []),
                "lofo": s.get("lofo", []),
            }
        if getattr(self, "unsupervised_", None) and "w" in self.unsupervised_:
            out["fit_mode"] = "unsupervised (silhouette objective)"
        return out

    def snapshot(self, df=None, name: str | None = None,
                 source: str | None = None, notes: str | None = None,
                 schema=None, profile: str | None = None,
                 aliases: dict | None = None) -> dict:
        """封成可存档、可跨批次对比的快照（见 hybridscope.monitor）。

        schema：本次分析用的字段映射；跨批次对比时用它判断哪些列可比较。
        aliases：canonical 名 → 建模列名（两者不同名时必填，否则声明落不到列上）。
        """
        from hybridscope.monitor import snapshot as _snap
        return _snap(self, df=df, name=name, source=source, notes=notes,
                     schema=schema, profile=profile, aliases=aliases)
