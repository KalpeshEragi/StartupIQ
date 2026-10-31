"""Task 6 market-pulse classifier: will the next 12 weeks be rising, flat or falling?

Label: mean of the next 12 weeks vs the last 12 weeks: above +5% rising, below -5% falling,
else flat. Features are known at the origin. Baselines: the recent slope, and what
happened in the same weeks last year. Series: the weekly panel (Rossmann stores, UCI products).
"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from startupiq.evaluate import report
from startupiq.evaluate.metrics import macro_f1, mean_sd
from startupiq.features import weekly as fw
from startupiq.features.splits import SEEDS, WEEK_HORIZON as H

THRESHOLD = 0.05
CLASSES = ["falling", "flat", "rising"]
FEATURES = ["growth_12", "slope_8", "last_year_change", "cv_12", "zero_share_12", "weekofyear", "dataset_code", "log_level"]


def _label(change: float) -> str:
    return "rising" if change > THRESHOLD else ("falling" if change < -THRESHOLD else "flat")


def _row(y: np.ndarray, o: int, week, ds: int) -> dict:
    last, prev = y[o - 12:o], y[o - 24:o - 12]
    x = np.arange(8)
    slope = np.polyfit(x, y[o - 8:o], 1)[0] if np.std(y[o - 8:o]) > 0 else 0.0
    ly_next, ly_last = y[o - fw.SEASON:o - fw.SEASON + H], y[o - fw.SEASON - 12:o - fw.SEASON]
    return {"growth_12": (last.mean() + 1) / (prev.mean() + 1) - 1, "slope_8": slope * 12 / (last.mean() + 1),
            "last_year_change": (ly_next.mean() + 1) / (ly_last.mean() + 1) - 1, "cv_12": last.std() / (last.mean() + 1),
            "zero_share_12": float(np.mean(last == 0)), "weekofyear": int(week.isocalendar()[1]), "dataset_code": ds,
            "log_level": np.log1p(last.mean())}


def build() -> pd.DataFrame:
    series, weeks = fw.panel()
    rows = []
    for sid, y in series.items():
        n, ds = len(y), 0 if sid.startswith("ross_") else 1
        test_o, val_o = n - H, n - 2 * H
        for o in range(fw.SEASON + 12, test_o + 1, 4):
            split = "test" if o == test_o else "val" if o == val_o else "train" if o + H <= val_o else None
            if split is None:
                continue
            r = _row(y, o, weeks[sid][o], ds)
            r.update(series=sid, origin=o, split=split, label=_label((y[o:o + H].mean() + 1) / (y[o - 12:o].mean() + 1) - 1))
            rows.append(r)
        # make sure the val and test origins exist even when not on the 4-week grid
        for o, split in ((val_o, "val"), (test_o, "test")):
            if (o - fw.SEASON - 12) % 4:
                r = _row(y, o, weeks[sid][o], ds)
                r.update(series=sid, origin=o, split=split, label=_label((y[o:o + H].mean() + 1) / (y[o - 12:o].mean() + 1) - 1))
                rows.append(r)
    return pd.DataFrame(rows)


def _models(seed):
    return {"Logistic regression": make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
            "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10, class_weight="balanced", n_jobs=-1, random_state=seed),
            "LightGBM": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight="balanced", subsample=0.8,
                                           subsample_freq=1, random_state=seed, verbose=-1)}


def _evaluate(d, train_splits, eval_split):
    tr, ev = d[d.split.isin(train_splits)], d[d.split == eval_split]
    res = [{"model": "Recent slope (8 weeks)", "f1": [macro_f1(ev.label, ev.slope_8.map(_label))], "acc": [accuracy_score(ev.label, ev.slope_8.map(_label))]},
           {"model": "Same weeks last year", "f1": [macro_f1(ev.label, ev.last_year_change.map(_label))], "acc": [accuracy_score(ev.label, ev.last_year_change.map(_label))]}]
    preds = {}
    for name in _models(0):
        f1s, accs = [], []
        for seed in (SEEDS[:1] if name.startswith("Logistic") else SEEDS):
            p = _models(seed)[name].fit(tr[FEATURES].fillna(-1) if name == "Random forest" else tr[FEATURES], tr.label).predict(
                ev[FEATURES].fillna(-1) if name == "Random forest" else ev[FEATURES])
            f1s.append(macro_f1(ev.label, p)); accs.append(accuracy_score(ev.label, p))
            preds.setdefault(name, p)
        res.append({"model": name, "f1": f1s, "acc": accs})
    table = pd.DataFrame([{"model": r["model"], "macro-F1": mean_sd(r["f1"]), "accuracy": mean_sd(r["acc"])} for r in res])
    return table, preds, ev


def run(final: bool = False) -> dict:
    t0 = time.time()
    d = build()
    counts = d.groupby("split").label.value_counts(normalize=True).round(3).unstack()
    table, preds, ev = _evaluate(d, ["train"], "val")
    cm = pd.DataFrame(confusion_matrix(ev.label, preds["LightGBM"], labels=CLASSES), index=[f"true {c}" for c in CLASSES], columns=[f"pred {c}" for c in CLASSES])
    out = {"validation": table.to_dict("records"), "rows": d.split.value_counts().to_dict()}
    sections = [("Data", f"Rows (series x origin): {out['rows']}. Label shares:\n\n" + report.table(counts.reset_index(), "{:.3f}")),
                ("Validation", report.table(table)),
                ("Confusion matrix (validation, LightGBM seed 0)", report.table(cm.reset_index().rename(columns={"index": ""})))]
    if final:
        ttable, _, _ = _evaluate(d, ["train", "val"], "test")
        out["test"] = ttable.to_dict("records")
        sections.append(("Test (scored once)", report.table(ttable)))
    out["seconds"] = round(time.time() - t0)
    report.write("regime", "Market-pulse (trend regime) classifier (Task 6)", sections, out)
    return out
