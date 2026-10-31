"""Task 6 forecast benchmark on Rossmann (daily, 42-day horizon, RMSPE on open days).

Validation (origin 2015-05-09) chooses settings. `final=True` refits on everything
before 2015-06-20 and scores the test window once. Candidates and decision rule:
devdocs/decisions/0005-forecaster.md.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from startupiq.evaluate import report
from startupiq.evaluate.metrics import coverage, mean_sd, rmspe
from startupiq.features import rossmann as fr
from startupiq.features.splits import ROSS_TEST_START, ROSS_VAL_START, SEEDS

LGB_GRID = [{"num_leaves": 31, "n_estimators": 600}, {"num_leaves": 127, "n_estimators": 600},
            {"num_leaves": 127, "n_estimators": 1500}, {"num_leaves": 255, "n_estimators": 1500}]
RIDGE_ALPHAS = [0.1, 1.0, 10.0]
RF_ROWS = 300_000


def _lgb(params: dict, seed: int, objective: str = "regression", alpha: float | None = None) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(objective=objective, alpha=alpha, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, min_child_samples=50, random_state=seed, verbose=-1, n_jobs=-1, **params)


def _ridge(alpha: float):
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), fr.NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), fr.CATEGORICAL + ["DayOfWeek", "month"]),
    ])
    return make_pipeline(pre, Ridge(alpha=alpha))


def _rf(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=100, max_features=0.5, min_samples_leaf=5, n_jobs=4, random_state=seed)  # 4 workers: each copies the data


def _codes(x: pd.DataFrame) -> pd.DataFrame:
    """Random forest needs numbers: categories become integer codes."""
    x = x.copy()
    for c in fr.CATEGORICAL:
        x[c] = x[c].cat.codes
    return x


def baselines(f: pd.DataFrame, origin: pd.Timestamp, ev: pd.DataFrame) -> dict[str, np.ndarray]:
    hist = f[(f.Date < origin) & f.log_sales.notna()]
    mean = hist.groupby("Store").Sales.mean()
    last4 = hist[hist.Date >= origin - pd.Timedelta(days=28)].groupby(["Store", "DayOfWeek"]).Sales.mean()
    wd = ev.set_index(["Store", "DayOfWeek"]).index.map(last4).to_numpy(dtype=float)
    ly = np.expm1(ev.lag_364.to_numpy())
    return {
        "Store mean": ev.Store.map(mean).to_numpy(),
        "Same weekday, last 4 weeks": wd,
        "Same day last year (rows with history only)": ly,
        "Same day last year, else same weekday 4 weeks": np.where(np.isfinite(ly), ly, wd),
    }


def _split(f: pd.DataFrame, origin: pd.Timestamp):
    train = f[(f.Date < origin) & f.log_sales.notna()]
    ev = f[(f.Date >= origin) & f.log_sales.notna()]
    return train, ev


def run(final: bool = False) -> dict:
    """Validation run; with final=True, score the test window using the settings recorded by the last validation run."""
    if final:
        return run_test()
    t0 = time.time()
    daily = pd.read_parquet("data/processed/rossmann_daily.parquet")
    out: dict = {"validation": {}, "test": {}}
    rows: list[dict] = []

    # ---------- validation: choose settings ----------
    fv = fr.build(daily, ROSS_VAL_START)
    tr, ev = _split(fv, ROSS_VAL_START)
    y_ev = ev.Sales.to_numpy()
    for name, p in baselines(fv, ROSS_VAL_START, ev).items():
        rows.append({"model": name, "split": "validation", "rmspe": rmspe(y_ev, p), "rows_scored": int(np.isfinite(p).sum())})

    ridge_scores = {}
    for a in RIDGE_ALPHAS:
        m = _ridge(a).fit(tr[fr.FEATURES], tr.log_sales)
        ridge_scores[a] = rmspe(y_ev, np.expm1(m.predict(ev[fr.FEATURES])))
    best_alpha = min(ridge_scores, key=ridge_scores.get)
    rows.append({"model": f"Ridge (alpha={best_alpha})", "split": "validation", "rmspe": ridge_scores[best_alpha], "rows_scored": len(ev)})

    rf_scores = []
    sample = tr.sample(min(RF_ROWS, len(tr)), random_state=0)
    for seed in SEEDS:
        m = _rf(seed).fit(_codes(sample[fr.FEATURES]), sample.log_sales)
        rf_scores.append(rmspe(y_ev, np.expm1(m.predict(_codes(ev[fr.FEATURES])))))
    rows.append({"model": f"Random forest ({RF_ROWS:,} rows)", "split": "validation", "rmspe": float(np.mean(rf_scores)),
                 "sd": float(np.std(rf_scores)), "rows_scored": len(ev)})

    grid = {}
    for i, params in enumerate(LGB_GRID):
        m = _lgb(params, 0).fit(tr[fr.FEATURES], tr.log_sales)
        grid[i] = rmspe(y_ev, np.expm1(m.predict(ev[fr.FEATURES])))
    best = LGB_GRID[min(grid, key=grid.get)]
    out["validation"]["lightgbm_grid"] = [{**p, "rmspe": grid[i]} for i, p in enumerate(LGB_GRID)]
    lgb_scores, models = [], []
    for seed in SEEDS:
        m = _lgb(best, seed).fit(tr[fr.FEATURES], tr.log_sales)
        lgb_scores.append(rmspe(y_ev, np.expm1(m.predict(ev[fr.FEATURES]))))
        models.append(m)
    rows.append({"model": f"LightGBM ({best['num_leaves']} leaves, {best['n_estimators']} trees)", "split": "validation",
                 "rmspe": float(np.mean(lgb_scores)), "sd": float(np.std(lgb_scores)), "rows_scored": len(ev)})

    # Interval: quantile LightGBM at 10% and 90%.
    lo = _lgb(best, 0, "quantile", 0.1).fit(tr[fr.FEATURES], tr.log_sales)
    hi = _lgb(best, 0, "quantile", 0.9).fit(tr[fr.FEATURES], tr.log_sales)
    out["validation"]["p10_p90_coverage"] = coverage(y_ev, np.expm1(lo.predict(ev[fr.FEATURES])), np.expm1(hi.predict(ev[fr.FEATURES])))

    # Feature importance and selection (Task 5): permutation importance on a validation sample.
    samp = ev.sample(20000, random_state=0)
    pi = permutation_importance(models[0], samp[fr.FEATURES], samp.log_sales, n_repeats=3, random_state=0,
                                scoring="neg_root_mean_squared_error", n_jobs=1)
    imp = pd.DataFrame({"feature": fr.FEATURES, "permutation_rmse_increase": pi.importances_mean,
                        "gain": models[0].booster_.feature_importance("gain") / models[0].booster_.feature_importance("gain").sum()})
    imp = imp.sort_values("permutation_rmse_increase", ascending=False).reset_index(drop=True)
    keep = imp[imp.permutation_rmse_increase > 0.001].feature.tolist()
    sel = _lgb(best, 0).fit(tr[keep], tr.log_sales)
    out["validation"]["selection"] = {"kept": keep, "rmspe_selected": rmspe(y_ev, np.expm1(sel.predict(ev[keep]))),
                                      "rmspe_all": lgb_scores[0]}

    # Error analysis on validation (best LightGBM, seed 0).
    ev = ev.assign(pred=np.expm1(models[0].predict(ev[fr.FEATURES])))
    ev["ape2"] = ((ev.Sales - ev.pred) / ev.Sales) ** 2
    by = lambda col: ev.groupby(col, observed=True).ape2.mean().pow(0.5).round(4).to_dict()  # noqa: E731
    worst = ev.groupby("Store").ape2.mean().pow(0.5).sort_values(ascending=False).head(10)
    out["validation"]["errors"] = {
        "by_store_type": by("StoreType"), "by_promo": by("Promo"), "by_weekday": by("DayOfWeek"),
        "by_2014_gap_store": by("store_has_2014_gap"), "lag_364_missing": by(ev.lag_364.isna().rename("lag_364_missing")),
        "worst_stores": worst.round(4).to_dict(),
        "worst_store_reasons": {int(s): _diagnose(fv, s) for s in worst.index[:5]},
    }

    res = pd.DataFrame(rows)
    res["rmspe ± sd"] = [mean_sd([r]) if pd.isna(sd) else f"{r:.4f} ± {sd:.4f}" for r, sd in zip(res.rmspe, res.get("sd", pd.Series(np.nan, index=res.index)))]
    out["table"] = res.to_dict("records")
    out["best_lightgbm"] = best
    out["best_ridge_alpha"] = best_alpha
    out["seconds"] = round(time.time() - t0)
    sections = [
        ("Results (lower is better)", report.table(res[["model", "split", "rmspe ± sd", "rows_scored"]])),
        ("LightGBM settings tried on validation", report.table(pd.DataFrame(out["validation"]["lightgbm_grid"]))),
        ("Interval", f"P10 to P90 coverage, validation: {out['validation']['p10_p90_coverage']:.3f} (target about 0.80)."),
        ("Feature importance (validation, LightGBM seed 0)", report.table(imp)),
        ("Feature selection", f"Kept {len(keep)} of {len(fr.FEATURES)} features with permutation importance > 0.001. "
                              f"Validation RMSPE {out['validation']['selection']['rmspe_selected']:.4f} with selection vs {lgb_scores[0]:.4f} with all."),
        ("Error analysis (validation)", "```\n" + "\n".join(f"{k}: {v}" for k, v in out["validation"]["errors"].items()) + "\n```"),
    ]
    report.write("forecast_rossmann", "Rossmann daily forecast benchmark (Task 6)", sections, out)
    return out


def _diagnose(f: pd.DataFrame, store: int) -> str:
    s = f[f.Store == store].sort_values("Date")
    closed_run = (s.Open == 0).astype(int).groupby((s.Open != 0).cumsum()).sum().max()
    gap = bool(s.store_has_2014_gap.iloc[0])
    return f"longest closed run {closed_run} days; 2014 gap store: {gap}; StoreType {s.StoreType.iloc[0]}"


def run_test() -> dict:
    """Refit every model on all data before 2015-06-20 with the settings chosen on validation; score the test window once.

    Settings come from docs/results/forecast_rossmann.json (the validation run), so the test never influences them.
    """
    t0 = time.time()
    chosen = json.loads(Path("docs/results/forecast_rossmann.json").read_text(encoding="utf-8"))
    best = chosen["best_lightgbm"]
    ridge_row = next(r["model"] for r in chosen["table"] if r["model"].startswith("Ridge"))
    best_alpha = chosen.get("best_ridge_alpha", float(ridge_row.split("=")[1].rstrip(")")))  # older runs recorded it only in the table
    daily = pd.read_parquet("data/processed/rossmann_daily.parquet")
    ft = fr.build(daily, ROSS_TEST_START)
    tr, te = _split(ft, ROSS_TEST_START)
    y = te.Sales.to_numpy()
    rows = [{"model": name, "rmspe": rmspe(y, p), "sd": np.nan, "rows_scored": int(np.isfinite(p).sum())}
            for name, p in baselines(ft, ROSS_TEST_START, te).items()]
    m = _ridge(best_alpha).fit(tr[fr.FEATURES], tr.log_sales)
    rows.append({"model": f"Ridge (alpha={best_alpha})", "rmspe": rmspe(y, np.expm1(m.predict(te[fr.FEATURES]))), "sd": np.nan, "rows_scored": len(te)})
    sample = tr.sample(min(RF_ROWS, len(tr)), random_state=0)
    s = [rmspe(y, np.expm1(_rf(seed).fit(_codes(sample[fr.FEATURES]), sample.log_sales).predict(_codes(te[fr.FEATURES])))) for seed in SEEDS]
    rows.append({"model": f"Random forest ({RF_ROWS:,} rows)", "rmspe": float(np.mean(s)), "sd": float(np.std(s)), "rows_scored": len(te)})
    s = [rmspe(y, np.expm1(_lgb(best, seed).fit(tr[fr.FEATURES], tr.log_sales).predict(te[fr.FEATURES]))) for seed in SEEDS]
    rows.append({"model": f"LightGBM ({best['num_leaves']} leaves, {best['n_estimators']} trees)", "rmspe": float(np.mean(s)), "sd": float(np.std(s)), "rows_scored": len(te)})
    lo = _lgb(best, 0, "quantile", 0.1).fit(tr[fr.FEATURES], tr.log_sales)
    hi = _lgb(best, 0, "quantile", 0.9).fit(tr[fr.FEATURES], tr.log_sales)
    cov = coverage(y, np.expm1(lo.predict(te[fr.FEATURES])), np.expm1(hi.predict(te[fr.FEATURES])))
    res = pd.DataFrame(rows)
    res["rmspe ± sd"] = [f"{r:.4f}" if pd.isna(sd) else f"{r:.4f} ± {sd:.4f}" for r, sd in zip(res.rmspe, res.sd)]
    out = {"table": res.to_dict("records"), "p10_p90_coverage": cov, "settings": {"lightgbm": best, "ridge_alpha": best_alpha},
           "seconds": round(time.time() - t0)}
    report.write("forecast_rossmann_test", "Rossmann daily forecast: test window 2015-06-20..07-31 (scored once)", [
        ("Settings", f"Taken from the validation run: LightGBM {best}, ridge alpha {best_alpha}. All models refit on data before 2015-06-20."),
        ("Test RMSPE (lower is better)", report.table(res[["model", "rmspe ± sd", "rows_scored"]])),
        ("Interval", f"LightGBM quantile P10 to P90 coverage on test: {cov:.3f} (target about 0.80)."),
    ], out)
    return out
