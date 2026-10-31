"""Wide layouts -> one row per (period, product, amount).

Two shapes are common in small-business sheets:
1. Periods across: | Product | Jan-24 | Feb-24 | ... |
2. Products across: | Date | Tea | Coffee | Snacks |   (no amount column, several numeric product columns)
"""

from __future__ import annotations

import pandas as pd

from startupiq.intake.columns import SYNONYMS, name_score, score_columns
from startupiq.intake.parsing import looks_like_period, to_dates, to_number

MIN_PERIOD_COLUMNS = 3
MIN_PRODUCT_COLUMNS = 3


def periods_across(df: pd.DataFrame) -> pd.DataFrame | None:
    periods = [c for c in df.columns if looks_like_period(c)]
    if len(periods) < MIN_PERIOD_COLUMNS:
        return None
    ids = [c for c in df.columns if c not in periods]
    long = df.melt(id_vars=ids, value_vars=periods, var_name="Period", value_name="Amount")
    long["Period"], _ = to_dates(long["Period"].astype(str))
    long["Amount"] = to_number(long["Amount"])
    product = next((c for c in ids if name_score(c, "product") > 0), ids[0] if ids else None)
    keep = ["Period", "Amount"] + ([product] if product else [])
    long = long[keep].rename(columns={product: "Product"} if product else {})
    if "Product" in long and long.Product.astype(str).str.lower().str.startswith("total").any():
        long = long[~long.Product.astype(str).str.lower().str.startswith("total")]
    return long


def products_across(df: pd.DataFrame, date_col: str) -> pd.DataFrame | None:
    s = score_columns(df)
    amount_named = (s[[f"name_{r}" for r in ("revenue", "units")]] > 0).any(axis=None)
    numeric = s[(s.numeric_share >= 0.8) & ~s.id_like & (s.index != date_col)]
    if amount_named or len(numeric) < MIN_PRODUCT_COLUMNS:
        return None
    if (numeric[[f"name_{r}" for r in SYNONYMS if r != "product"]] > 0).any(axis=None):
        return None  # columns are prices, costs or stock, not products
    long = df.melt(id_vars=[date_col], value_vars=list(numeric.index), var_name="Product", value_name="Amount")
    long["Amount"] = to_number(long["Amount"])
    return long.rename(columns={date_col: "Date"})
