"""Rossmann cleaning rules. One function per rule; each logs its effect.

Decisions: devdocs/decisions/0010-cleaning-rules.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from startupiq.clean.log import CleaningLog

RAW = Path("data/raw/rossmann_hf")
GAP_START, GAP_END = pd.Timestamp("2014-07-01"), pd.Timestamp("2014-12-31")
OUTLIER_Z = 5.0


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(RAW / "train.csv", low_memory=False, dtype={"StateHoliday": str}, parse_dates=["Date"])
    store = pd.read_csv(RAW / "store.csv")
    return train, store


def drop_duplicate_store_days(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    out = df.drop_duplicates(["Store", "Date"])
    log.record("drop_duplicate_store_days", len(df), len(out), "One row per store and date")
    return out


def mark_zero_sales_on_open_days(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Open but zero sales is almost always a recording error, not zero demand."""
    bad = (df.Open == 1) & (df.Sales == 0)
    out = df.copy()
    out["sales_missing"] = bad
    out.loc[bad, "Sales"] = np.nan
    log.record("mark_zero_sales_on_open_days", len(df), len(out), f"{int(bad.sum())} store-days set to missing (kept as rows)")
    return out


def flag_outliers(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Robust z-score of log sales within each store; flagged, never dropped."""
    out = df.copy()
    ok = (out.Open == 1) & (out.Sales > 0)
    ls = np.log(out.loc[ok, "Sales"])
    med = ls.groupby(out.loc[ok, "Store"]).transform("median")
    mad = (ls - med).abs().groupby(out.loc[ok, "Store"]).transform("median")
    z = (ls - med) / (1.4826 * mad)
    out["outlier"] = False
    out.loc[z.index, "outlier"] = z.abs() > OUTLIER_Z
    log.record("flag_outliers", len(df), len(out), f"{int(out.outlier.sum())} store-days with abs(robust z) > {OUTLIER_Z} on log sales flagged")
    return out


def flag_known_gap(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """180 stores have no rows from 2014-07-01 to 2014-12-31 (missing in the source, not closures)."""
    days = df.groupby("Store").Date.nunique()
    total = df.Date.nunique()
    in_gap = df[(df.Date >= GAP_START) & (df.Date <= GAP_END)].Store.unique()
    gap_stores = sorted(set(days[days < total].index) - set(in_gap))
    out = df.copy()
    out["store_has_2014_gap"] = out.Store.isin(gap_stores)
    log.record("flag_known_gap", len(df), len(out), f"{len(gap_stores)} stores missing 2014-07-01..2014-12-31; flagged, gap left empty (not filled)")
    return out


def clean_store_table(store: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    out = store.copy()
    out["competition_distance_missing"] = out.CompetitionDistance.isna()
    out["competition_open_unknown"] = out.CompetitionOpenSinceYear.isna()
    # Promo2 fields are empty because the store is not in Promo2, not because data is missing.
    out.loc[out.Promo2 == 0, ["Promo2SinceWeek", "Promo2SinceYear"]] = 0
    out["PromoInterval"] = out.PromoInterval.fillna("none")
    log.record("clean_store_table", len(store), len(out),
               f"{int(out.competition_distance_missing.sum())} stores without competition distance and "
               f"{int(out.competition_open_unknown.sum())} without opening date flagged (left NaN for train-only imputation); "
               "Promo2 blanks set to 0/'none' where store has no Promo2")
    return out


def weekly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["week"] = d.Date.dt.to_period("W-SUN").dt.end_time.dt.normalize()
    g = d.groupby(["Store", "week"]).agg(
        sales=("Sales", "sum"), customers=("Customers", "sum"),
        open_days=("Open", "sum"), promo_days=("Promo", "sum"),
        school_holiday_days=("SchoolHoliday", "sum"),
        state_holiday_days=("StateHoliday", lambda s: int((s != "0").sum())),
        days=("Date", "nunique"), missing_days=("sales_missing", "sum"))
    return g.reset_index()


def run(out_dir: Path = Path("data/processed")) -> CleaningLog:
    log = CleaningLog("rossmann")
    train, store = load()
    log.record("load", len(train), len(train), f"{train.Store.nunique()} stores, {train.Date.min().date()} to {train.Date.max().date()}")
    df = drop_duplicate_store_days(train, log)
    df = mark_zero_sales_on_open_days(df, log)
    df = flag_outliers(df, log)
    df = flag_known_gap(df, log)
    st = clean_store_table(store, log)
    df = df.merge(st, on="Store", how="left", validate="many_to_one")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "rossmann_daily.parquet", index=False)
    wk = weekly(df)
    # Weeks cut by the start or end of the data are partial; drop them from the weekly table.
    full = wk[wk.days == 7]
    log.record("weekly_aggregate", len(wk), len(full), "Store-weeks; partial first/last weeks and gap-edge weeks dropped")
    full.to_parquet(out_dir / "rossmann_weekly.parquet", index=False)
    log.save(out_dir)
    return log
