"""Parse the numbers and dates people actually type into spreadsheets."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

CURRENCY = re.compile(r"(₹|rs\.?|inr|usd|\$|£|€|gbp|eur|/-)", re.IGNORECASE)
EXCEL_EPOCH = pd.Timestamp("1899-12-30")
SERIAL_RANGE = (20000, 60000)  # Excel day numbers for 1954..2064
MONTH_ONLY = re.compile(r"^(?:[A-Za-z]{3,9}[\s\-/']*\d{2,4}|\d{4}[\-/]\d{1,2}|\d{1,2}[\-/]\d{4})$")
SPLIT = re.compile(r"[\-/.]")


def to_number(values: pd.Series) -> pd.Series:
    """'₹1,25,000.50', 'Rs. 450/-', '(300)', ' 12 ' -> floats. Unparseable -> NaN."""
    if pd.api.types.is_numeric_dtype(values):
        return values.astype(float)
    text = values.astype("string").str.strip()
    negative = text.str.match(r"^\(.*\)$").fillna(False)
    text = text.str.replace(CURRENCY, "", regex=True).str.replace(r"[,\s()]", "", regex=True)
    text = text.replace({"": pd.NA, "-": pd.NA, "—": pd.NA})
    out = pd.to_numeric(text, errors="coerce").astype(float)
    return out.where(~negative, -out)


def numeric_share(values: pd.Series) -> float:
    present = values.dropna()
    if len(present) == 0:
        return 0.0
    return float(to_number(present).notna().mean())


def day_order(text: pd.Series) -> tuple[bool, bool]:
    """Decide day-first vs month-first from the values. Returns (dayfirst, ambiguous)."""
    parts = text.str.split(SPLIT, expand=True)
    if parts.shape[1] < 3:
        return True, False
    first = pd.to_numeric(parts[0], errors="coerce")
    second = pd.to_numeric(parts[1], errors="coerce")
    if (first > 12).any():
        return True, False
    if (second > 12).any():
        return False, False
    return True, True  # every value fits both ways: default to day-first (India, UK) and ask


def to_dates(values: pd.Series) -> tuple[pd.Series, str]:
    """Parse a column into timestamps. Returns (dates, how) where how names the rule used."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values), "datetime"
    nums = pd.to_numeric(values, errors="coerce")
    present = nums.dropna()
    if len(present) and present.between(*SERIAL_RANGE).mean() > 0.9 and (present == present.round()).mean() > 0.9:
        return EXCEL_EPOCH + pd.to_timedelta(nums, unit="D"), "excel_serial"
    text = values.astype("string").str.strip()
    sample = text.dropna()
    if len(sample) == 0:
        return pd.Series(pd.NaT, index=values.index), "empty"
    if sample.str.match(r"^\d{4}-\d{1,2}-\d{1,2}").mean() > 0.5:
        return pd.to_datetime(text, errors="coerce", format="ISO8601"), "iso"
    if sample.str.match(MONTH_ONLY).mean() > 0.8:
        parsed = pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=True)
        return parsed.dt.to_period("M").dt.to_timestamp(), "month_only"
    dayfirst, ambiguous = day_order(sample.str.split(r"\s", n=1).str[0])
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=dayfirst, format="mixed")
    how = "day_first" if dayfirst else "month_first"
    return parsed, how + ("_ambiguous" if ambiguous else "")


def date_share(values: pd.Series) -> float:
    """Share of non-empty values that parse as plausible dates (1990..2100)."""
    present = values.dropna()
    if len(present) == 0:
        return 0.0
    if pd.api.types.is_numeric_dtype(present) and not present.between(*SERIAL_RANGE).all():
        return 0.0  # plain numbers are not dates unless they look like Excel serials
    parsed, _ = to_dates(present)
    ok = parsed.notna() & (parsed.dt.year >= 1990) & (parsed.dt.year <= 2100)
    return float(ok.mean())


def looks_like_period(label: str) -> bool:
    """Is a column header a date or month, as in wide layouts ('Jan-2024', '2024-03', '01/03/2024')?"""
    s = str(label).strip()
    if not re.search(r"\d", s):
        return False
    if MONTH_ONLY.match(s) or re.match(r"^\d{4}-\d{1,2}(-\d{1,2})?( 00:00:00)?$", s) or re.match(r"^\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}$", s):
        parsed = pd.to_datetime(pd.Series([s]), errors="coerce", format="mixed", dayfirst=True).iloc[0]
        return pd.notna(parsed) and 1990 <= parsed.year <= 2100
    return False


def robust_z(values: np.ndarray) -> np.ndarray:
    """Median/MAD z-score; robust to the outliers it is looking for."""
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad == 0:
        return np.zeros_like(values, dtype=float)
    return (values - med) / (1.4826 * mad)
