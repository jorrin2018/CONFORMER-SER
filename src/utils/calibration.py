"""Calibracion de predictores probabilisticos: ECE, Brier, NLL y calibradores simples.

Lo usa scripts/expert_gate_offline.py para la fusion guiada por incertidumbre de los
expertos congelados (propuesta de la asesora, 04/11/2026: Expected Calibration Error y
Brier score).

Convenciones
------------
- P: (N, C) probabilidades; y: (N,) etiquetas enteras.
- ECE de top-label (Guo et al. 2017): bins de igual ancho sobre la confianza maxima,
  ECE = sum_b (n_b / N) * |acc_b - conf_b|.
- Brier multiclase: media sobre muestras de sum_c (p_c - 1[y = c])^2. Rango [0, 2].
- Todos los calibradores se ajustan en VALIDACION y se aplican a test: ajustarlos en
  test daria una calibracion perfecta y falsa.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def ece(P: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = P.max(1)
    acc = (P.argmax(1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(total)


def brier(P: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(P.shape[1])[y]
    return float(((P - onehot) ** 2).sum(1).mean())


def brier_per_class(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(C,) Brier restringido a las muestras de cada clase verdadera."""
    onehot = np.eye(P.shape[1])[y]
    err = ((P - onehot) ** 2).sum(1)
    return np.array([err[y == c].mean() if (y == c).any() else np.nan
                     for c in range(P.shape[1])])


def nll(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(P[np.arange(len(y)), y], EPS, 1.0)).mean())


# ------------------------------------------------------------- calibradores
def apply_temperature(P: np.ndarray, T: float) -> np.ndarray:
    """Escalado de temperatura sobre probabilidades (equivale a logits / T)."""
    logit = np.log(np.clip(P, EPS, 1.0)) / T
    logit -= logit.max(1, keepdims=True)
    e = np.exp(logit)
    return e / e.sum(1, keepdims=True)


def fit_temperature(P_val: np.ndarray, y_val: np.ndarray,
                    grid: np.ndarray | None = None) -> float:
    """T que minimiza la NLL en validacion (busqueda en rejilla logaritmica, robusta)."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(10.0), 200)) if grid is None else grid
    losses = [nll(apply_temperature(P_val, T), y_val) for T in grid]
    return float(grid[int(np.argmin(losses))])


class ReliabilityMap:
    """Confianza -> exactitud esperada, estimada en validacion por bins (la curva de
    fiabilidad que mide la ECE). Bins con pocas muestras se encogen hacia la exactitud
    global para no pesar ruido."""

    def __init__(self, n_bins: int = 10, prior_strength: float = 10.0):
        self.n_bins, self.k = n_bins, prior_strength
        self.edges = np.linspace(0.0, 1.0, n_bins + 1)
        self.acc = None

    def fit(self, P: np.ndarray, y: np.ndarray) -> "ReliabilityMap":
        conf = P.max(1)
        hit = (P.argmax(1) == y).astype(float)
        base = hit.mean()
        idx = np.clip(np.digitize(conf, self.edges[1:-1]), 0, self.n_bins - 1)
        self.acc = np.array([(hit[idx == b].sum() + self.k * base) / ((idx == b).sum() + self.k)
                             for b in range(self.n_bins)])
        return self

    def __call__(self, P: np.ndarray) -> np.ndarray:
        idx = np.clip(np.digitize(P.max(1), self.edges[1:-1]), 0, self.n_bins - 1)
        return self.acc[idx]
