"""Weekly series panel and origin-based features for the global weekly forecaster.

Every feature is computed from values strictly before the forecast origin o and
scaled by the series' recent level, so one model can serve a cafe and a drugstore.
Target: y[o + h - 1] / scale for h = 1..12.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from startupiq.features.splits import WEEK_HORIZON

LAGS = 12
SEASON = 52
FEATURES = [f"lag_{i}" for i in range(1, LAGS + 1)] + [
    "mean_4", "mean_12", "mean_52", "std_12", "zero_share_12", "seasonal_same_week", "seasonal_ratio",
    "trend_12", "h", "weekofyear", "dataset_code"]


def panel(min_active_weeks: int = 52) -> tuple[dict[str, np.ndarray], dict[str, pd.DatetimeIndex]]:
    """series_id -> weekly values (regular, zero-filled where the business traded nothing), and their week index."""
    out: dict[str, np.ndarray] = {}
    weeks: dict[str, pd.DatetimeIndex] = {}
    rw = pd.read_parquet("data/processed/rossmann_weekly.parquet")
    daily_flags = pd.read_parquet("data/processed/rossmann_daily.parquet", columns=["Store", "store_has_2014_gap"]).drop_duplicates("Store")
    keep = set(daily_flags[~daily_flags.store_has_2014_gap].Store)
    idx = pd.date_range(rw.week.min(), rw.week.max(), freq="W-SUN")
    for store, g in rw[rw.Store.isin(keep)].groupby("Store"):
        s = g.set_index("week").sales.reindex(idx)
        if s.isna().any():
            continue
        out[f"ross_{store}"] = s.to_numpy(float); weeks[f"ross_{store}"] = idx
    up = pd.read_parquet("data/processed/uci_product_weekly.parquet")
    idx = pd.date_range(up.week.min(), up.week.max(), freq="W-SUN")
    active = up[up.units > 0].groupby("StockCode").week.nunique()
    for code in active[active >= min_active_weeks].index:
        s = up[up.StockCode == code].set_index("week").revenue.reindex(idx, fill_value=0.0)
        out[f"uci_{code}"] = s.clip(lower=0).to_numpy(float); weeks[f"uci_{code}"] = idx
    return out, weeks


def origin_features(y: np.ndarray, o: int, weeks: pd.DatetimeIndex, dataset_code: int, horizon: int = WEEK_HORIZON) -> pd.DataFrame:
    """One row per horizon step for forecasting y[o:o+horizon] from y[:o]. Missing history -> NaN."""
    hist = y[:o]
    scale = (np.mean(hist[-12:]) if len(hist) else 0.0) + 1.0
    def tail(n):  # noqa: E301
        return hist[-n:] if len(hist) >= n else np.full(n, np.nan)
    lags = tail(LAGS)[::-1] / scale
    base = {f"lag_{i + 1}": lags[i] for i in range(LAGS)}
    base.update(mean_4=np.mean(tail(4)) / scale, mean_12=np.mean(tail(12)) / scale,
                mean_52=np.mean(tail(SEASON)) / scale, std_12=np.std(tail(12)) / scale,
                zero_share_12=np.mean(tail(12) == 0) if len(hist) >= 12 else np.nan,
                trend_12=(np.mean(tail(6)) - np.mean(tail(12)[:6])) / scale,
                dataset_code=dataset_code)
    last_year_window = y[o - SEASON - 12:o - SEASON] if o - SEASON - 12 >= 0 else None
    rows = []
    for h in range(1, horizon + 1):
        t = o + h - 1
        r = dict(base, h=h, weekofyear=int(weeks[t].isocalendar()[1]) if t < len(weeks) else np.nan)
        ly = y[t - SEASON] if t - SEASON >= 0 and t - SEASON < o else np.nan
        r["seasonal_same_week"] = ly / scale
        r["seasonal_ratio"] = (ly / (np.mean(last_year_window) + 1.0)) if last_year_window is not None and np.isfinite(ly) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def scale_of(y: np.ndarray, o: int) -> float:
    return float(np.mean(y[max(0, o - 12):o])) + 1.0
