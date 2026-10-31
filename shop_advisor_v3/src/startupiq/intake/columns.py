"""Decide which column plays which role, using the header name AND the values.

Each role gets a confidence in 0..1. Below CONFIRM_BELOW the UI asks the user to
confirm. Why rules and not a learned classifier: devdocs/decisions/0011-intake-design.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from startupiq.intake.parsing import date_share, numeric_share, to_number

CONFIRM_BELOW = 0.75
NAME_MIN = 0.5  # a name match weakened below this ("freight_value", "GST Amount") does not count

# Normalised name fragments per role. Hindi words cover common Indian shop registers.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "date": ("date", "day", "dt", "invoicedate", "orderdate", "billdate", "txndate", "transactiondate", "timestamp",
             "createdat", "period", "month", "week", "तारीख", "दिनांक", "tarikh"),
    "revenue": ("sales", "revenue", "amount", "total", "netamount", "grossamount", "value", "turnover", "billamount",
                "invoicevalue", "income", "takings", "collection", "receipts", "bikri", "बिक्री", "राशि", "रकम", "kul"),
    "units": ("qty", "quantity", "units", "pcs", "pieces", "count", "noofitems", "itemssold", "मात्रा", "नग"),
    "price": ("price", "rate", "unitprice", "mrp", "sellingprice", "दर", "मूल्य", "kimat"),
    "product": ("product", "item", "sku", "stockcode", "description", "productname", "itemname", "article",
                "उत्पाद", "सामान", "वस्तु"),
    "cost": ("cost", "cogs", "purchaseprice", "buyingprice", "लागत"),
    "stock": ("stock", "inventory", "closingstock", "onhand"),
    "channel": ("channel", "platform", "marketplace", "source", "store", "outlet"),
}
# Words that make a numeric column unlikely to be the sale amount or quantity.
NOT_AMOUNT = ("id", "no", "number", "phone", "mobile", "pin", "zip", "code", "gst", "tax", "discount", "year",
              "freight", "shipping", "delivery", "fee", "commission", "weight", "length", "height", "width")
# A file often has several dates; the sale date is the one to use.
DATE_PREFER = ("order", "purchase", "invoice", "bill", "sale", "txn", "transaction")
DATE_AVOID = ("ship", "deliver", "estimat", "approv", "limit", "due", "expir", "birth", "dob", "carrier", "review", "answer")


def _words(column: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", str(column))}


def norm(name: str) -> str:
    return re.sub(r"[^0-9a-zऀ-ॿ]", "", str(name).lower())


def name_score(column: str, role: str) -> float:
    n = norm(column)
    if not n:
        return 0.0
    best = 0.0
    for syn in SYNONYMS[role]:
        if n == syn:
            return 1.0
        if syn in n and len(syn) >= 3:
            best = max(best, 0.7)
    if role in ("revenue", "units", "price", "cost", "stock") and best > 0:
        if _words(column) & set(NOT_AMOUNT):  # "Invoice No", "Customer ID", "GST Amount", "freight_value"
            best *= 0.4
    if role == "date" and best > 0:
        if any(a in n for a in DATE_AVOID):  # shipping_limit_date, estimated_delivery_date
            best *= 0.4
        elif best < 1.0 and any(p in n for p in DATE_PREFER):  # order_purchase_timestamp
            best = 0.9
    return best


def is_id_like(values: pd.Series) -> bool:
    nums = to_number(values.dropna())
    if len(nums) < 5 or nums.isna().any():
        return False
    return bool(nums.is_unique and (nums == nums.round()).all() and nums.is_monotonic_increasing)


def is_year_like(values: pd.Series) -> bool:
    nums = to_number(values.dropna())
    return len(nums) > 0 and bool(nums.between(1990, 2100).all() and (nums == nums.round()).all())


@dataclass
class Assignment:
    column_map: dict[str, str]
    confidence: dict[str, float]

    @property
    def needs_confirmation(self) -> list[str]:
        return [r for r, c in self.confidence.items() if c < CONFIRM_BELOW]


def score_columns(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column with name scores per role and content statistics."""
    rows = []
    for col in df.columns:
        v = df[col]
        present = v.dropna()
        rows.append({
            "column": col,
            "date_share": date_share(present.head(500)),
            "numeric_share": numeric_share(present.head(500)),
            "distinct_ratio": present.nunique() / max(len(present), 1),
            "fill": len(present) / max(len(v), 1),
            "id_like": is_id_like(present.head(500)),
            "year_like": is_year_like(present.head(500)),
            **{f"name_{r}": name_score(col, r) for r in SYNONYMS},
        })
    return pd.DataFrame(rows).set_index("column")


def assign_roles(df: pd.DataFrame, overrides: dict[str, str] | None = None) -> Assignment:
    s = score_columns(df)
    used: set[str] = set()
    cmap: dict[str, str] = {}
    conf: dict[str, float] = {}

    def take(role: str, col: str, c: float) -> None:
        cmap[role], conf[role] = col, round(min(c, 1.0), 2)
        used.add(col)

    for role, col in (overrides or {}).items():
        if col in df.columns:
            take(role, col, 1.0)

    # Date: content must parse; the name only breaks ties and lifts confidence.
    if "date" not in cmap:
        cand = s[(s.date_share >= 0.8) & ~s.index.isin(used)]
        if len(cand):
            score = 0.6 * cand.date_share + 0.4 * cand.name_date
            best = score.idxmax()
            take("date", best, 0.55 + 0.45 * (0.5 * cand.loc[best, "date_share"] + 0.5 * cand.loc[best, "name_date"]))

    numeric = s[(s.numeric_share >= 0.8) & ~s.id_like & ~s.year_like & (s.date_share < 0.8)]
    for role in ("revenue", "units", "price", "cost", "stock"):
        if role in cmap:
            continue
        cand = numeric[~numeric.index.isin(used) & (numeric[f"name_{role}"] >= NAME_MIN)]
        if len(cand):
            best = cand[f"name_{role}"].idxmax()
            take(role, best, 0.6 + 0.4 * cand.loc[best, f"name_{role}"] * cand.loc[best, "numeric_share"])

    # Unnamed numbers (e.g. Column2): if no money or quantity column was named, guess the money column
    # by content: the numeric column with the largest typical value. Always asks for confirmation.
    if not {"revenue", "units", "price"} & cmap.keys():
        cand = numeric[~numeric.index.isin(used)]
        if len(cand) == 1 or (len(cand) > 1 and not (cand[[f"name_{r}" for r in SYNONYMS]] > 0).any(axis=None)):
            med = {c: float(to_number(df[c]).abs().median()) for c in cand.index}
            if med:
                take("revenue", max(med, key=med.get), 0.5)

    # Product: text column, repeated values, not the date.
    if "product" not in cmap:
        text = s[(s.numeric_share < 0.5) & (s.date_share < 0.5) & ~s.index.isin(used) & (s.fill > 0.5)]
        named = text[text.name_product > 0]
        if len(named):
            best = named.name_product.idxmax()
            take("product", best, 0.6 + 0.4 * named.loc[best, "name_product"])
        else:
            repeated = text[(text.distinct_ratio < 0.5) & (text.distinct_ratio > 0)]
            if len(repeated) == 1:
                take("product", repeated.index[0], 0.5)

    for role in ("channel",):
        if role not in cmap:
            cand = s[(s[f"name_{role}"] > 0) & ~s.index.isin(used)]
            if len(cand):
                best = cand[f"name_{role}"].idxmax()
                take(role, best, 0.6 + 0.4 * cand.loc[best, f"name_{role}"])
    return Assignment(cmap, conf)
