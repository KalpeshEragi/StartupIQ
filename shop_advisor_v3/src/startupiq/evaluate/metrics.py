"""Metrics used across experiments (devdocs/decisions/0009-metrics.md)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def rmspe(actual, pred) -> float:
    """Root mean squared percentage error over actual > 0 (Rossmann's official metric)."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    m = (a > 0) & np.isfinite(p)
    return float(np.sqrt(np.mean(((a[m] - p[m]) / a[m]) ** 2)))


def mase_scale(history, m: int) -> float:
    """Mean absolute m-step seasonal difference of the training history."""
    h = np.asarray(history, float)
    if len(h) <= m:
        return np.nan
    return float(np.mean(np.abs(h[m:] - h[:-m])))


def mase(actual, pred, scale: float) -> float:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    if not np.isfinite(scale) or scale == 0:
        return np.nan
    return float(np.mean(np.abs(a - p)) / scale)


def smape(actual, pred) -> float:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    d = np.abs(a) + np.abs(p)
    m = d > 0
    return float(np.mean(2 * np.abs(a[m] - p[m]) / d[m]))


def coverage(actual, lo, hi) -> float:
    a = np.asarray(actual, float)
    return float(np.mean((a >= np.asarray(lo, float)) & (a <= np.asarray(hi, float))))


def macro_f1(y, pred) -> float:
    return float(f1_score(y, pred, average="macro", zero_division=0))


def brier_multiclass(y, proba, classes) -> float:
    """Mean over rows of the summed squared error between one-hot truth and predicted probabilities."""
    onehot = (np.asarray(y)[:, None] == np.asarray(classes)[None, :]).astype(float)
    return float(np.mean(np.sum((np.asarray(proba) - onehot) ** 2, axis=1)))


def mean_sd(values) -> str:
    v = np.asarray(values, float)
    return f"{v.mean():.4f} ± {v.std(ddof=0):.4f}" if len(v) > 1 else f"{v.mean():.4f}"
