# -*- coding: utf-8 -*-
"""PSO v <- omega·v + c1·r1·(pbest - x) + c2·r2·(gbest - x) x <- x + v"""

from __future__ import annotations

import numpy as np


def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = np.asarray(x, dtype=float) / max(temp, 1e-9)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class PSO:
    def __init__(self, dim: int, n_particles: int = 30, iters: int = 50,
                 w_inertia: float = 0.9, w_end: float = 0.4,
                 c1: float = 2.0, c2: float = 2.0, v_max: float = 4.0,
                 seed: int = 0, verbose: bool = False):
        self.dim = dim
        self.P = n_particles
        self.T = iters
        self.w0, self.w1 = w_inertia, w_end
        self.c1, self.c2 = c1, c2
        self.v_max = v_max
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

    def _update(self, omega: float, x, v, pbest, gbest):
        r1 = self.rng.random(x.shape)
        r2 = self.rng.random(x.shape)
        v = omega * v + self.c1 * r1 * (pbest - x) + self.c2 * r2 * (gbest - x)  # 式(14)
        v = np.clip(v, -self.v_max, self.v_max)
        return x + v, v

    def optimize(self, fitness, encoding: str = "weight", n_items: int | None = None,
                 x0: np.ndarray | None = None):
        """最小化 fitness。返回 (best_x, best_f, history)。"""
        if encoding == "weight":
            return self._optimize_weight(fitness, x0)
        return self._optimize_neighbor(fitness, n_items)

    def _optimize_weight(self, fitness, x0=None):
        m = self.dim
        x = self.rng.normal(0.0, 1.0, size=(self.P, m)) if x0 is None else \
            np.tile(np.asarray(x0, dtype=float), (self.P, 1)) + self.rng.normal(0, 0.1, (self.P, m))
        v = self.rng.normal(0.0, 0.5, size=(self.P, m))
        pbest = x.copy()
        pbest_f = np.array([fitness(_softmax(xi)) for xi in x])
        g = int(np.argmin(pbest_f))
        gbest = pbest[g].copy()
        gbest_f = float(pbest_f[g])
        hist = [gbest_f]

        for t in range(self.T):
            omega = self.w0 + (self.w1 - self.w0) * (t / max(self.T - 1, 1))
            for i in range(self.P):
                x[i], v[i] = self._update(omega, x[i], v[i], pbest[i], gbest)
                f = fitness(_softmax(x[i]))
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbest[i] = x[i].copy()
                    if f < gbest_f:
                        gbest_f = float(f)
                        gbest = x[i].copy()
            hist.append(gbest_f)
            if self.verbose and (t + 1) % 10 == 0:
                print(f"  [PSO] iter {t + 1}/{self.T}  best={gbest_f:.4f}")
        return _softmax(gbest), gbest_f, hist

    def _optimize_neighbor(self, fitness, n_items):
        k = self.dim
        n = int(n_items)
        x = self.rng.integers(0, n, size=(self.P, k)).astype(float)
        v = self.rng.normal(0.0, 1.0, size=(self.P, k))

        def decode(xi):
            return np.clip(np.rint(xi), 0, n - 1).astype(int)

        pbest = x.copy()
        pbest_f = np.array([fitness(decode(xi)) for xi in x])
        g = int(np.argmin(pbest_f))
        gbest = pbest[g].copy()
        gbest_f = float(pbest_f[g])
        hist = [gbest_f]

        for t in range(self.T):
            omega = self.w0 + (self.w1 - self.w0) * (t / max(self.T - 1, 1))
            for i in range(self.P):
                x[i], v[i] = self._update(omega, x[i], v[i], pbest[i], gbest)
                x[i] = np.clip(x[i], 0, n - 1)
                f = fitness(decode(x[i]))
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbest[i] = x[i].copy()
                    if f < gbest_f:
                        gbest_f = float(f)
                        gbest = x[i].copy()
            hist.append(gbest_f)
        return decode(gbest), gbest_f, hist
