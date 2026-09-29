"""Task 6 evaluation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def score(y_true, proba, threshold=0.5) -> dict:
    """Metrics with 'acquired' (1) as the positive class."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def error_rate_by(frame: pd.DataFrame, col: str, min_n=40, top=10) -> pd.DataFrame:
    """Error / FP / FN rate per group. frame needs columns: target, pred."""
    f = frame.assign(err=(frame.target != frame.pred).astype(int),
                     fp=((frame.pred == 1) & (frame.target == 0)).astype(int),
                     fn=((frame.pred == 0) & (frame.target == 1)).astype(int))
    g = f.groupby(col, dropna=False).agg(n=("err", "size"), error_rate=("err", "mean"),
                                         fp_rate=("fp", "mean"), fn_rate=("fn", "mean"),
                                         acquired_share=("target", "mean"))
    return g[g.n >= min_n].sort_values("n", ascending=False).head(top).round(3)
