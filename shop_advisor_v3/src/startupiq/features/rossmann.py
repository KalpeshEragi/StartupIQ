"""Rossmann daily features, computed relative to a forecast origin.

Rule: a feature for date d may only use information known at the origin. Sales
information is therefore used either from the training window (store encodings)
or with a lag of at least the 42-day horizon (lag_364, roll_42_28). Calendar,
promotion, holiday and opening schedules are planned in advance and allowed.
Customers is never used (it is only known after the day). Leakage checklist:
devdocs/leakage_checklist.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from startupiq.features.splits import ROSS_HORIZON_DAYS

MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec"}
CATEGORICAL = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]
NUMERIC = ["DayOfWeek", "day", "month", "year", "weekofyear", "dayofyear", "Promo", "SchoolHoliday",
           "open_prev", "open_next", "days_to_holiday", "days_since_holiday", "log_comp_dist", "comp_months",
           "promo2_active", "lag_364", "roll_42_28", "store_mean", "store_dow_mean", "store_promo_lift", "store_month_dev"]
FEATURES = NUMERIC + CATEGORICAL


def _calendar(d: pd.DataFrame) -> pd.DataFrame:
    d["day"], d["month"], d["year"] = d.Date.dt.day, d.Date.dt.month, d.Date.dt.year
    d["weekofyear"] = d.Date.dt.isocalendar().week.astype(int)
    d["dayofyear"] = d.Date.dt.dayofyear
    return d


def _store_schedule(d: pd.DataFrame) -> pd.DataFrame:
    """Opening the day before/after and distance to state holidays: planned in advance."""
    d = d.sort_values(["Store", "Date"])
    g = d.groupby("Store")
    d["open_prev"] = g.Open.shift(1)
    d["open_next"] = g.Open.shift(-1)
    hol = d.Date.where(d.StateHoliday != "0")
    d["days_since_holiday"] = (d.Date - hol.groupby(d.Store).ffill()).dt.days.clip(upper=30)
    d["days_to_holiday"] = (hol.groupby(d.Store).bfill() - d.Date).dt.days.clip(upper=30)
    return d


def _store_static(d: pd.DataFrame) -> pd.DataFrame:
    d["log_comp_dist"] = np.log1p(d.CompetitionDistance)
    comp_year = d.CompetitionOpenSinceYear.where(d.CompetitionOpenSinceYear >= 1980)  # 1900 and 1961 are entry errors
    d["comp_months"] = (d.year - comp_year) * 12 + (d.month - d.CompetitionOpenSinceMonth)
    since = pd.to_datetime(d.Promo2SinceYear.astype("Int64").astype(str) + "-01-01", errors="coerce", format="%Y-%m-%d") + \
        pd.to_timedelta((d.Promo2SinceWeek.fillna(1) - 1) * 7, unit="D")
    in_interval = [MONTH_ABBR[m] in str(p).split(",") for m, p in zip(d.month, d.PromoInterval)]
    d["promo2_active"] = ((d.Promo2 == 1) & (d.Date >= since) & np.array(in_interval)).astype(int)
    return d


def _lagged_sales(d: pd.DataFrame) -> pd.DataFrame:
    """Sales seen at least one horizon ago, so the feature exists for every forecast day."""
    d = d.sort_values(["Store", "Date"])
    logs = d.set_index("Date").groupby("Store").log_sales
    full = []
    for store, s in logs:
        s = s.asfreq("D")  # calendar gaps (the 2014 hole) become NaN
        lag364 = s.shift(364)
        roll = s.shift(ROSS_HORIZON_DAYS).rolling(28, min_periods=10).mean()
        full.append(pd.DataFrame({"Store": store, "Date": s.index, "lag_364": lag364.values, "roll_42_28": roll.values}))
    return d.merge(pd.concat(full), on=["Store", "Date"], how="left")


def _encodings(d: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Per-store averages from the training window only (Date < origin)."""
    t = d[(d.Date < origin) & d.log_sales.notna()]
    store_mean = t.groupby("Store").log_sales.mean().rename("store_mean")
    dow = t.groupby(["Store", "DayOfWeek"]).log_sales.mean().rename("store_dow_mean")
    promo = t.groupby(["Store", "Promo"]).log_sales.mean().unstack()
    lift = (promo[1] - promo[0]).rename("store_promo_lift")
    month = (t.groupby(["Store", "month"]).log_sales.mean() - t.groupby("Store").log_sales.mean()).rename("store_month_dev")
    d = d.merge(store_mean, on="Store", how="left").merge(dow, on=["Store", "DayOfWeek"], how="left")
    d = d.merge(lift, on="Store", how="left").merge(month, on=["Store", "month"], how="left")
    return d


def build(daily: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Feature table for every store-day up to origin + horizon. `log_sales` is the target (NaN when closed)."""
    end = origin + pd.Timedelta(days=ROSS_HORIZON_DAYS - 1)
    d = _store_schedule(daily.copy())  # schedules use Open/StateHoliday only, which are planned ahead
    d = d[d.Date <= end].copy()
    d["log_sales"] = np.where((d.Open == 1) & (d.Sales > 0), np.log1p(d.Sales), np.nan)
    d = _calendar(d)
    d = _store_static(d)
    d = _lagged_sales(d)
    d = _encodings(d, origin)
    for c in CATEGORICAL:
        d[c] = d[c].astype(str).astype("category")
    return d.sort_values(["Date", "Store"]).reset_index(drop=True)
