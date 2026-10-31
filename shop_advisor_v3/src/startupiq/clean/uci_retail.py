"""UCI Online Retail II cleaning rules. One function per rule; each logs its effect.

Decisions: devdocs/decisions/0010-cleaning-rules.md.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from startupiq.clean.log import CleaningLog

RAW_ZIP = Path("data/raw/uci_retail/online_retail_ii.zip")
CACHE = Path("data/interim/uci_retail_raw.parquet")
PRODUCT_CODE = r"^\d{5}"  # real products start with five digits; POST, DOT, M, BANK CHARGES etc. do not


def load() -> pd.DataFrame:
    """Reading the 45 MB workbook takes about 2 minutes, so the raw concat is cached as parquet."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    with zipfile.ZipFile(RAW_ZIP) as z, z.open("online_retail_II.xlsx") as f:
        sheets = pd.read_excel(f, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df = df.rename(columns={"Customer ID": "CustomerID"})
    df["Description"] = df.Description.astype("string")  # a few descriptions are numbers in Excel
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    return df


def drop_exact_duplicates(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    out = df.drop_duplicates()
    log.record("drop_exact_duplicates", len(df), len(out), "Identical lines (same invoice, product, qty, time, price)")
    return out


def drop_non_products(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    is_product = df.StockCode.astype(str).str.match(PRODUCT_CODE)
    top = df.loc[~is_product, "StockCode"].value_counts().head(6).to_dict()
    out = df[is_product]
    log.record("drop_non_products", len(df), len(out), f"Postage, fees and manual adjustments removed; most common: {top}")
    return out


def net_off_cancellations(df: pd.DataFrame, log: CleaningLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A cancellation line (invoice starting with C) removes the latest earlier sale with the
    same customer, product and quantity. Unmatched cancellations are partial returns: kept
    in a separate returns table, not subtracted from demand."""
    cancel = df.Invoice.astype(str).str.startswith("C")
    sales, cancels = df[~cancel].copy(), df[cancel].copy()
    key = ["CustomerID", "StockCode", "q"]
    sales["q"], cancels["q"] = sales.Quantity, -cancels.Quantity
    counts = cancels.dropna(subset=["CustomerID"]).groupby(key).size().rename("n_cancel")
    sales = sales.sort_values("InvoiceDate")
    sales["rank_from_end"] = sales.groupby(key, dropna=False).cumcount(ascending=False)
    sales = sales.join(counts, on=key)
    matched = sales.n_cancel.notna() & (sales.rank_from_end < sales.n_cancel)
    kept = sales[~matched].drop(columns=["q", "rank_from_end", "n_cancel"])
    returns = cancels.drop(columns=["q"])
    log.record("net_off_cancellations", len(df), len(kept),
               f"{int(cancel.sum()):,} cancellation lines removed; {int(matched.sum()):,} matching sales lines removed with them")
    return kept, returns


def drop_non_positive(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    out = df[(df.Quantity > 0) & (df.Price > 0)]
    log.record("drop_non_positive", len(df), len(out), "Remaining zero/negative quantity or price lines (damages, write-offs)")
    return out


def flag_missing_customer(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    out = df.assign(customer_missing=df.CustomerID.isna())
    log.record("flag_missing_customer", len(df), len(out), f"{int(out.customer_missing.sum()):,} lines without customer ID kept (not needed for sales)")
    return out


def weekly_tables(lines: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = lines.assign(revenue=lines.Quantity * lines.Price,
                     week=lines.InvoiceDate.dt.to_period("W-SUN").dt.end_time.dt.normalize())
    product = d.groupby(["StockCode", "week"]).agg(units=("Quantity", "sum"), revenue=("revenue", "sum"),
                                                   price=("Price", "median"), orders=("Invoice", "nunique")).reset_index()
    total = d.groupby("week").agg(revenue=("revenue", "sum"), units=("Quantity", "sum"), orders=("Invoice", "nunique")).reset_index()
    # Fill weeks with no trading (Christmas closures) with zero so the calendar is regular.
    full = pd.DataFrame({"week": pd.date_range(total.week.min(), total.week.max(), freq="W-SUN")})
    total = full.merge(total, on="week", how="left").fillna({"revenue": 0, "units": 0, "orders": 0})
    return product, total


def run(out_dir: Path = Path("data/processed")) -> CleaningLog:
    log = CleaningLog("uci_retail")
    df = load()
    log.record("load", len(df), len(df), f"{df.StockCode.nunique():,} stock codes, {df.InvoiceDate.min().date()} to {df.InvoiceDate.max().date()}")
    df = drop_exact_duplicates(df, log)
    df = drop_non_products(df, log)
    df, returns = net_off_cancellations(df, log)
    df = drop_non_positive(df, log)
    df = flag_missing_customer(df, log)
    product, total = weekly_tables(df)
    # First week starts on a Tuesday and last week ends on a Friday: both partial.
    first, last = total.week.min(), total.week.max()
    weeks_before = len(total)
    total = total[(total.week > first) & (total.week < last)]
    product = product[(product.week > first) & (product.week < last)]
    log.record("weekly_aggregate", weeks_before, len(total),
               f"Rows are weeks: partial first and last week dropped; {product.StockCode.nunique():,} products in product-week table")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "uci_lines.parquet", index=False)
    returns.to_parquet(out_dir / "uci_returns.parquet", index=False)
    product.to_parquet(out_dir / "uci_product_weekly.parquet", index=False)
    total.to_parquet(out_dir / "uci_total_weekly.parquet", index=False)
    log.save(out_dir)
    return log
