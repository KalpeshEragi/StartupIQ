"""Fixed train / validation / test split and the statistical baselines every model must beat.

    python -m market.evaluate             (from ml/, after market.index)

Target: the weekly relative market interest index per sector (100 = a typical market), 19 series.
Split, by time only (no shuffling, so no model ever sees the future):
  - test       = the last 12 weeks. Locked: used once, at the very end.
  - validation = the 12 weeks before that. Used to pick models and settings.
  - train      = everything earlier (from January 2023).
Every model forecasts 1-12 weeks ahead from the last week it is allowed to see.

Baselines:
  - naive:          next weeks = this week
  - seasonal naive: next weeks = the same weeks a year earlier
  - ETS:            exponential smoothing with a damped trend (statsmodels)
Ranges (P10 / P90) for the baselines come from their own past errors at each horizon on the training data.

Metrics:
  - MASE: mean absolute error divided by the average week-to-week change in training. Below 1 = better than
    "no change" on average; the main comparison number.
  - MAE in index points, and coverage: share of actual values inside P10-P90 (ideal about 80%).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WEEKLY = ROOT / "data" / "processed" / "market" / "sectors_weekly.parquet"
OUT = ROOT / "models" / "runs" / "market_baselines.json"
H = 12          # forecast horizon in weeks
SEASON = 52


def load() -> pd.DataFrame:
    """Weeks x sectors, relative index."""
    w = pd.read_parquet(WEEKLY).pivot(index="week_end", columns="sector", values="index").sort_index()
    return w.dropna()


def split(w: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"train": w.iloc[:-2 * H], "val": w.iloc[-2 * H:-H], "test": w.iloc[-H:]}


def naive(hist: np.ndarray) -> np.ndarray:
    return np.repeat(hist[-1], H)


def seasonal_naive(hist: np.ndarray) -> np.ndarray:
    return hist[-SEASON:][:H] if len(hist) >= SEASON else naive(hist)


def ets(hist: np.ndarray) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(hist, trend="add", damped_trend=True, seasonal=None,
                                   initialization_method="estimated").fit()
    return np.asarray(fit.forecast(H))


METHODS = {"naive": naive, "seasonal_naive": seasonal_naive, "ets": ets}


def past_errors(method, hist: np.ndarray, origins: int = 30) -> np.ndarray:
    """Errors (actual - forecast) per horizon from rolling origins inside the history: shape (origins, H)."""
    errs = []
    for end in range(len(hist) - H - origins, len(hist) - H):
        if end < SEASON + 8:
            continue
        errs.append(hist[end:end + H] - method(hist[:end]))
    return np.array(errs)


def forecast(method, hist: np.ndarray) -> dict[str, np.ndarray]:
    p50 = method(hist)
    e = past_errors(method, hist)
    if len(e):
        return {"p10": p50 + np.quantile(e, 0.1, axis=0), "p50": p50, "p90": p50 + np.quantile(e, 0.9, axis=0)}
    return {"p10": p50, "p50": p50, "p90": p50}


def scores(actual: np.ndarray, f: dict[str, np.ndarray], scale: float) -> dict[str, float]:
    err = np.abs(actual - f["p50"])
    inside = (actual >= f["p10"]) & (actual <= f["p90"])
    return {"mase": float(err.mean() / scale), "mae": float(err.mean()), "coverage": float(inside.mean()),
            "mase_1_4": float(err[:4].mean() / scale), "mase_5_8": float(err[4:8].mean() / scale),
            "mase_9_12": float(err[8:].mean() / scale)}


def run(w: pd.DataFrame, period: str) -> dict:
    """Forecast the 12 weeks of `period` ('val' or 'test') from everything before it, for every sector."""
    parts = split(w)
    hist_df = parts["train"] if period == "val" else pd.concat([parts["train"], parts["val"]])
    target = parts[period]
    out: dict = {}
    for name, method in METHODS.items():
        per = {}
        for sector in w.columns:
            hist = hist_df[sector].to_numpy()
            scale = float(np.mean(np.abs(np.diff(parts["train"][sector].to_numpy()))))
            per[sector] = scores(target[sector].to_numpy(), forecast(method, hist), scale)
        keys = next(iter(per.values())).keys()
        out[name] = {"overall": {k: float(np.mean([p[k] for p in per.values()])) for k in keys}, "sectors": per}
    return out


def main() -> None:
    w = load()
    parts = split(w)
    info = {p: f"{d.index.min():%Y-%m-%d} to {d.index.max():%Y-%m-%d} ({len(d)} weeks)" for p, d in parts.items()}
    results = {"split": info, "sectors": list(w.columns), "val": run(w, "val"), "test": run(w, "test")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    for p, d in info.items():
        print(f"{p:>5}: {d}")
    for period in ("val", "test"):
        print(f"\n{period.upper()}  (19 sectors, 1-12 weeks ahead)")
        print(f"{'method':<16}{'MASE':>7}{'1-4w':>7}{'5-8w':>7}{'9-12w':>7}{'MAE':>7}{'P10-P90 cover':>15}")
        for name, r in results[period].items():
            o = r["overall"]
            print(f"{name:<16}{o['mase']:>7.2f}{o['mase_1_4']:>7.2f}{o['mase_5_8']:>7.2f}{o['mase_9_12']:>7.2f}"
                  f"{o['mae']:>7.1f}{o['coverage']:>14.0%}")
    print(f"\nSaved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
