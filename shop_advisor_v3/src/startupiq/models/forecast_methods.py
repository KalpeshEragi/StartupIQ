"""Per-series forecasting methods and the Auto selector (decision 0012, section 2).

Used both by the weekly benchmark (models/forecast_weekly.py) and by the product
(models/forecast_baseline.py), so what we measured is exactly what founders get.
Weekly series: season 52, horizon 12. Monthly series: season 12, horizon 3.
"""

from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

LEVEL_RATIO_CAP = (0.25, 4.0)  # a product that sold nothing last year must not get a 100x "growth" multiplier


def naive(hist, h=12, season=52):
    return np.repeat(hist[-1], h)


def moving_average(hist, h=12, season=52):
    return np.repeat(np.mean(hist[-4:]), h)


def seasonal_naive(hist, h=12, season=52):
    if len(hist) < season:
        return np.full(h, np.nan)
    return np.array([hist[len(hist) - season + (k % season)] for k in range(h)])


def seasonal_naive_level(hist, h=12, season=52):
    """Last year's same periods, scaled by how the recent level compares with the same stretch last year."""
    window = min(12, season)
    if len(hist) < season + window:
        return np.full(h, np.nan)
    ratio = np.clip((np.mean(hist[-window:]) + 1) / (np.mean(hist[-season - window:-season]) + 1), *LEVEL_RATIO_CAP)
    return seasonal_naive(hist, h, season) * ratio


def ets(hist, h=12, season=52):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = ExponentialSmoothing(np.log1p(hist), trend="add", damped_trend=True, initialization_method="estimated").fit()
            return np.expm1(m.forecast(h))
        except Exception:
            return moving_average(hist, h, season)


METHODS = {"Naive (last week)": naive, "Moving average (4 weeks)": moving_average, "Seasonal naive (52 weeks)": seasonal_naive,
           "Seasonal naive, level-adjusted": seasonal_naive_level, "ETS (damped trend, log)": ets}
AUTO = "Auto: best method on the series' own last 12 weeks"


def auto(hist, h=12, season=52) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Backtest every method on the last h periods of the history, keep the winner, and build an
    80% band from that backtest's errors (additive 10th and 90th percentiles). Returns (p50, p10, p90, method)."""
    hist = np.asarray(hist, float)
    inner, check = hist[:-h], hist[-h:]
    best, best_err, best_res = "Moving average (4 weeks)", np.inf, None
    for name, fn in METHODS.items():
        if len(inner) < 4:
            break
        p = fn(inner, h, season)
        if not np.isfinite(p).all():
            continue
        err = np.mean(np.abs(check - p))
        if err < best_err:
            best, best_err, best_res = name, err, check - p
    fc = METHODS[best](hist, h, season)
    if not np.isfinite(fc).all():
        best, fc = "Moving average (4 weeks)", moving_average(hist, h, season)
    if best_res is not None:
        lo_q, hi_q = np.quantile(best_res, [0.1, 0.9])
    else:
        lo_q, hi_q = -0.5 * np.mean(hist[-4:]), 0.5 * np.mean(hist[-4:])
    return np.clip(fc, 0, None), np.clip(fc + lo_q, 0, None), np.clip(fc + hi_q, 0, None), best
