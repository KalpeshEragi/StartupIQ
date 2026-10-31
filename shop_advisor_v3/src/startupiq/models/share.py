"""Task 6 market-share model on Olist: next month's share of category revenue.

Only categories with 5+ active sellers in month t (a market, not a near-monopoly).
Baseline: persistence (next share = current share). Metric: MAE in percentage points,
overall and by seller size band.
"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from startupiq.evaluate import report
from startupiq.evaluate.metrics import mean_sd
from startupiq.features import olist as fo
from startupiq.features.splits import SEEDS

MIN_SELLERS = 5


def _models(seed: int) -> dict:
    pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), fo.NUMERIC),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), fo.CATEGORICAL)])
    tree_pre = ColumnTransformer([("num", SimpleImputer(strategy="median", add_indicator=True), fo.NUMERIC),
                                  ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), fo.CATEGORICAL)])
    return {
        "Ridge": make_pipeline(pre, Ridge(alpha=1.0)),
        "Random forest": make_pipeline(tree_pre, RandomForestRegressor(n_estimators=300, min_samples_leaf=10, n_jobs=-1, random_state=seed)),
        "LightGBM (L1 loss)": lgb.LGBMRegressor(objective="l1", n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1),
    }


def _mae_pp(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))) * 100)


def _evaluate(rows, train_splits, eval_split):
    tr, ev = rows[rows.split.isin(train_splits)], rows[rows.split == eval_split]
    res = [{"model": "Persistence (share stays the same)", "mae": [_mae_pp(ev.share_next, ev.share)], "pred": ev.share.to_numpy()}]
    for name in _models(0):
        maes, first = [], None
        for seed in (SEEDS[:1] if name == "Ridge" else SEEDS):
            m = _models(seed)[name].fit(tr[fo.FEATURES], tr.share_next)
            p = np.clip(m.predict(ev[fo.FEATURES]), 0, 1)
            maes.append(_mae_pp(ev.share_next, p))
            first = p if first is None else first
        res.append({"model": name, "mae": maes, "pred": first})
    return res, ev


def _table(res, ev) -> pd.DataFrame:
    band = pd.qcut(ev.log_revenue, 3, labels=["small", "medium", "large"])
    rows = []
    for r in res:
        row = {"model": r["model"], "MAE (pct points)": mean_sd(r["mae"])}
        for b in ["small", "medium", "large"]:
            m = (band == b).to_numpy()
            row[f"MAE {b} sellers"] = f"{_mae_pp(ev.share_next[m], r['pred'][m]):.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def run(final: bool = False) -> dict:
    t0 = time.time()
    rows = fo.build()
    rows = rows[(rows.category_sellers >= MIN_SELLERS) & rows.share_next.notna()].reset_index(drop=True)
    out = {"rows": rows.split.value_counts().to_dict(), "mean_share_pct": round(float(rows.share.mean() * 100), 3)}
    res, ev = _evaluate(rows, ["train"], "val")
    out["validation"] = _table(res, ev).to_dict("records")
    sections = [
        ("Data", f"Seller-category-months in categories with {MIN_SELLERS}+ active sellers: {out['rows']}. Mean share {out['mean_share_pct']}%."),
        ("Validation (MAE in percentage points; lower is better)", report.table(_table(res, ev))),
    ]
    if final:
        tres, te = _evaluate(rows, ["train", "val"], "test")
        out["test"] = _table(tres, te).to_dict("records")
        sections.append(("Test (refit on train + validation, scored once)", report.table(_table(tres, te))))
    out["seconds"] = round(time.time() - t0)
    report.write("share", "Market-share model (Task 6)", sections, out)
    return out
