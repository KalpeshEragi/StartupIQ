"""Weekly benchmark: 12-week forecasts for Rossmann stores and UCI products (founder-like series).

Per-series methods (naive, moving average, seasonal naive, ETS) against one global
LightGBM trained on many series. Metric: MASE (seasonal, m=52), pooled per dataset.
Also measures error against history length, to check the data tiers (decision 0004).
"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from startupiq.evaluate import report
from startupiq.evaluate.metrics import coverage, mase, mase_scale, mean_sd, smape
from startupiq.features import weekly as fw
from startupiq.features.splits import SEEDS, WEEK_HORIZON as H
from startupiq.models.forecast_methods import AUTO, METHODS, auto, ets, moving_average, naive

TRAIN_ORIGIN_STEP = 2
HISTORY_LENGTHS = [8, 13, 26, 39, 52]


PER_SERIES = METHODS  # shared with the product forecaster (models/forecast_methods.py)


def _dataset(sid: str) -> int:
    return 0 if sid.startswith("ross_") else 1


def _training_rows(series, weeks, last_target_exclusive: int | dict) -> pd.DataFrame:
    rows = []
    for sid, y in series.items():
        stop = last_target_exclusive[sid] if isinstance(last_target_exclusive, dict) else last_target_exclusive
        for o in range(fw.SEASON + 12, stop - H + 1, TRAIN_ORIGIN_STEP):
            f = fw.origin_features(y, o, weeks[sid], _dataset(sid))
            # log1p of the scaled value: a sparse product's 500x jump must not dominate the squared loss for every series
            f["target"] = np.log1p(y[o:o + H] / fw.scale_of(y, o))
            rows.append(f)
    return pd.concat(rows, ignore_index=True)


def _lgb(seed, objective="regression", alpha=None):
    return lgb.LGBMRegressor(objective=objective, alpha=alpha, n_estimators=600, learning_rate=0.03, num_leaves=63,
                             min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             random_state=seed, verbose=-1, n_jobs=-1)


def evaluate_origin(series, weeks, origins: dict[str, int], seeds=SEEDS) -> tuple[pd.DataFrame, dict]:
    """Score every method on y[o:o+12] for each series' origin o. Returns per-series errors and extras."""
    train = _training_rows(series, weeks, origins)
    feats = {sid: fw.origin_features(y, origins[sid], weeks[sid], _dataset(sid)) for sid, y in series.items()}
    X = pd.concat(feats.values(), ignore_index=True)[fw.FEATURES]
    scales = np.repeat([fw.scale_of(y, origins[sid]) for sid, y in series.items()], H)
    preds_lgb = []
    for seed in seeds:
        m = _lgb(seed).fit(train[fw.FEATURES], train.target)
        preds_lgb.append(np.clip(np.expm1(m.predict(X)), 0, None) * scales)
    lo = np.clip(np.expm1(_lgb(0, "quantile", 0.1).fit(train[fw.FEATURES], train.target).predict(X)), 0, None) * scales
    hi = np.clip(np.expm1(_lgb(0, "quantile", 0.9).fit(train[fw.FEATURES], train.target).predict(X)), 0, None) * scales

    rows, cov_actual, cov_lo, cov_hi, auto_lo, auto_hi, picks = [], [], [], [], [], [], []
    for i, (sid, y) in enumerate(series.items()):
        o = origins[sid]
        hist, actual = y[:o], y[o:o + H]
        scale = mase_scale(hist, fw.SEASON)
        r = {"series": sid, "dataset": "Rossmann stores" if _dataset(sid) == 0 else "UCI products"}
        for name, fn in PER_SERIES.items():
            p = fn(hist)
            r[name] = mase(actual, p, scale) if np.isfinite(p).all() else np.nan
            r[name + " sMAPE"] = smape(actual, p) if np.isfinite(p).all() else np.nan
        p, alo, ahi, pick = auto(hist)
        r[AUTO] = mase(actual, p, scale)
        auto_lo.append(alo); auto_hi.append(ahi); picks.append(pick)
        for k, p in enumerate(preds_lgb):
            r[f"LightGBM global seed{k}"] = mase(actual, p[i * H:(i + 1) * H], scale)
        r["LightGBM global sMAPE"] = smape(actual, preds_lgb[0][i * H:(i + 1) * H])
        rows.append(r)
        cov_actual.append(actual); cov_lo.append(lo[i * H:(i + 1) * H]); cov_hi.append(hi[i * H:(i + 1) * H])
    res = pd.DataFrame(rows)
    res["auto_pick"] = picks
    extra = {"coverage": {}, "auto_coverage": {}, "auto_picks": res.groupby("dataset").auto_pick.value_counts(normalize=True).round(3).to_dict()}
    a, l, h = np.concatenate(cov_actual), np.concatenate(cov_lo), np.concatenate(cov_hi)
    al, ah = np.concatenate(auto_lo), np.concatenate(auto_hi)
    for ds in res.dataset.unique():
        mask = np.repeat((res.dataset == ds).to_numpy(), H)
        extra["coverage"][ds] = coverage(a[mask], l[mask], h[mask])
        extra["auto_coverage"][ds] = coverage(a[mask], al[mask], ah[mask])
    return res, extra


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    out = []
    lgb_cols = [c for c in res.columns if c.startswith("LightGBM global seed")]
    for ds, g in list(res.groupby("dataset")) + [("Pooled", res)]:
        g = g[np.isfinite(g[list(PER_SERIES) + [AUTO]].to_numpy()).all(axis=1) & np.isfinite(g[lgb_cols].to_numpy()).all(axis=1)]
        row = {"dataset": ds, "series": len(g)}
        for name in list(PER_SERIES) + [AUTO]:
            row[name] = f"{g[name].mean():.3f}"
        row["LightGBM global"] = mean_sd([g[c].mean() for c in lgb_cols])
        out.append(row)
    return pd.DataFrame(out)


def tier_check(series, weeks, origins) -> pd.DataFrame:
    """MASE when only the last N weeks are available (scale always from full history, so numbers compare)."""
    rows = []
    for n in HISTORY_LENGTHS + ["all"]:
        r = {"history_weeks": n}
        vals = {name: [] for name in ["Moving average (4 weeks)", "Naive (last week)", "ETS (damped trend, log)", "Store/series mean",
                                      "Seasonal naive (52 weeks)", "Seasonal naive, level-adjusted"]}
        for sid, y in series.items():
            if not sid.startswith("ross_"):
                continue
            o = origins[sid]
            hist = y[:o] if n == "all" else y[o - n:o]
            scale = mase_scale(y[:o], fw.SEASON)
            actual = y[o:o + H]
            vals["Moving average (4 weeks)"].append(mase(actual, moving_average(hist), scale))
            vals["Naive (last week)"].append(mase(actual, naive(hist), scale))
            vals["ETS (damped trend, log)"].append(mase(actual, ets(hist), scale) if len(hist) >= 10 else np.nan)
            vals["Store/series mean"].append(mase(actual, np.repeat(np.mean(hist), H), scale))
            for name in ("Seasonal naive (52 weeks)", "Seasonal naive, level-adjusted"):
                p = PER_SERIES[name](hist)
                vals[name].append(mase(actual, p, scale) if np.isfinite(p).all() else np.nan)
        for k, v in vals.items():
            r[k] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def _r(d: dict) -> str:
    return ", ".join(f"{k} {v:.3f}" for k, v in d.items())


def run(final: bool = False) -> dict:
    """Validation run (writes forecast_weekly.md); final=True scores only the test origin (forecast_weekly_test.md)."""
    t0 = time.time()
    series, weeks = fw.panel()
    test_origin = {sid: len(y) - H for sid, y in series.items()}
    val_origin = {sid: o - H for sid, o in test_origin.items()}
    out: dict = {"series": {"Rossmann stores": sum(s.startswith("ross_") for s in series), "UCI products": sum(s.startswith("uci_") for s in series)}}
    if final:
        te, te_extra = evaluate_origin(series, weeks, test_origin)
        out["test"] = {"summary": summarise(te).to_dict("records"), "coverage": te_extra["coverage"], "auto_coverage": te_extra["auto_coverage"],
                       "auto_picks": {str(k): v for k, v in te_extra["auto_picks"].items()}}
        out["seconds"] = round(time.time() - t0)
        report.write("forecast_weekly_test", "Weekly forecast benchmark: test origin (scored once)", [
            ("Test MASE (12 weeks before the end; lower is better)", report.table(summarise(te))),
            ("Test P10 to P90 coverage (target about 0.80)", f"LightGBM quantile: {_r(te_extra['coverage'])}; Auto: {_r(te_extra['auto_coverage'])}"),
            ("Which method Auto picked (test)", str(te_extra["auto_picks"])),
        ], out)
        return out

    val, val_extra = evaluate_origin(series, weeks, val_origin)
    out["validation"] = {"summary": summarise(val).to_dict("records"), "coverage": val_extra["coverage"],
                         "auto_coverage": val_extra["auto_coverage"], "auto_picks": {str(k): v for k, v in val_extra["auto_picks"].items()}}
    tiers = tier_check(series, weeks, val_origin)
    out["tier_check_validation"] = tiers.to_dict("records")
    sections = [
        ("Data", f"{out['series']['Rossmann stores']} Rossmann stores (no 2014 gap) and {out['series']['UCI products']} UCI products with 52+ active weeks. "
                 "Horizon 12 weeks. Validation origin is 24 weeks before the end; test origin 12 weeks before the end."),
        ("Validation MASE (mean over series; lower is better; below 1 beats seasonal naive's in-sample error)", report.table(summarise(val))),
        ("Validation P10 to P90 coverage (target about 0.80)", f"LightGBM quantile: {_r(val_extra['coverage'])}; Auto (backtest errors): {_r(val_extra['auto_coverage'])}"),
        ("Which method Auto picked (validation)", str(val_extra["auto_picks"])),
        ("Tier check: MASE vs weeks of history (Rossmann stores, validation origin)", report.table(tiers, "{:.3f}")),
    ]
    out["seconds"] = round(time.time() - t0)
    report.write("forecast_weekly", "Weekly forecast benchmark (Task 6)", sections, out)
    return out
