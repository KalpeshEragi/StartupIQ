"""Task 4 preprocessing: raw Crunchbase investments_VC.csv -> cleaned dataset.

Deterministic and reproducible: `python Task4/run_task4.py` rebuilds
data/processed/startups_clean.csv from data/raw/investments_VC.csv.
The raw file is never modified.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FUNDING_TYPE_COLS = [
    "seed", "venture", "equity_crowdfunding", "undisclosed", "convertible_note",
    "debt_financing", "angel", "grant", "private_equity", "post_ipo_equity",
    "post_ipo_debt", "secondary_market", "product_crowdfunding",
]
ROUND_COLS = [f"round_{c}" for c in "ABCDEFGH"]
DATE_COLS = ["founded_at", "first_funding_at", "last_funding_at"]


def load_raw(path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")
    df.columns = df.columns.str.strip()  # 'market' and 'funding_total_usd' have leading spaces
    return df


def parse_money(s: pd.Series) -> pd.Series:
    """'17,50,000' (Indian digit grouping) -> 1750000.0 ; '-' -> NaN (unknown, not zero)."""
    cleaned = s.astype("string").str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log: dict = {"raw_rows": len(raw), "raw_cols": raw.shape[1]}
    df = raw.copy()

    # 1. whitespace / empty strings in text columns -> proper missing
    text_cols = df.select_dtypes(include="object").columns
    for c in text_cols:
        df[c] = df[c].astype("string").str.strip().replace("", pd.NA)

    # 2. completely blank rows (the "duplicates" reported by df.duplicated())
    blank = df.isna().all(axis=1)
    log["blank_rows_dropped"] = int(blank.sum())
    df = df[~blank]
    log["exact_duplicates_after_blank_removal"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # 3. duplicate companies (same permalink): keep the most complete record
    df = df.assign(_filled=df.notna().sum(axis=1))
    dup = df.permalink.duplicated(keep=False)
    log["duplicate_permalink_rows"] = int(dup.sum())
    df = (df.sort_values("_filled", ascending=False, kind="stable")
            .drop_duplicates("permalink", keep="first")
            .sort_index().drop(columns="_filled"))

    # 4. money stored as text
    df["funding_total_usd"] = parse_money(df["funding_total_usd"])
    log["funding_total_unknown"] = int(df.funding_total_usd.isna().sum())

    # funding_total == sum(type columns) exactly wherever it is known. Where the total is
    # unknown ('-') every type column is 0, i.e. "unknown", not "raised nothing"
    # (funding_rounds >= 1 for every company). Restore them to missing.
    unknown_funding = df.funding_total_usd.isna()
    df.loc[unknown_funding, FUNDING_TYPE_COLS] = np.nan
    log["funding_type_zeros_reset_to_missing_rows"] = int(unknown_funding.sum())

    # 5. dates
    for c in DATE_COLS:
        parsed = pd.to_datetime(df[c], errors="coerce")
        log[f"{c}_unparsable"] = int((parsed.isna() & df[c].notna()).sum())
        df[c] = parsed
    log["founded_before_1900"] = int((df.founded_at < "1900-01-01").sum())
    log["first_funding_before_founding_over_1y"] = int(
        ((df.founded_at - df.first_funding_at).dt.days > 365).sum())
    log["last_funding_before_first_funding"] = int((df.last_funding_at < df.first_funding_at).sum())

    # validity flags (values are kept, so later stages decide how to treat them)
    df["flag_funding_before_founding"] = (
        (df.founded_at - df.first_funding_at).dt.days > 365).astype("int8")
    df["flag_founded_before_1900"] = (df.founded_at < "1900-01-01").astype("int8")

    # 6. categories
    df["status"] = df["status"].str.lower()
    df["market"] = df["market"].str.strip()
    df["country_code"] = df["country_code"].str.upper()
    df["state_code"] = df["state_code"].str.upper()
    log["state_code_present_outside_usa"] = int(
        (df.state_code.notna() & (df.country_code != "USA")).sum())  # state_code exists only for USA and CAN

    # 7. skew: log transform of the money columns (scaling itself is fitted on train in Task 5)
    df["funding_total_usd_log1p"] = np.log1p(df.funding_total_usd)

    # 8. numeric types
    df["funding_rounds"] = df["funding_rounds"].astype("Int64")
    df["founded_year"] = df["founded_year"].astype("Int64")

    df = df.reset_index(drop=True)
    log["clean_rows"] = len(df)
    log["clean_cols"] = df.shape[1]
    log["status_counts"] = df.status.value_counts(dropna=False).rename(str).to_dict()
    return df, log

