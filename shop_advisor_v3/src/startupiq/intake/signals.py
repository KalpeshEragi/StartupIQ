"""Per-day (or per-week) register signals for the business-type classifier: sales, and when the file has them,
orders (bills), items sold and returns. Optional columns are found by name only; anything unclear is left out,
because the classifier works without them.
"""
from __future__ import annotations

import pandas as pd

from startupiq.intake.columns import norm
from startupiq.intake.parsing import to_dates, to_number

EXTRA: dict[str, tuple[str, ...]] = {
    "orders": ("orders", "order", "bills", "bill", "noofbills", "billcount", "invoices", "transactions", "txns", "customers", "footfall"),
    "items": ("items", "itemssold", "noofitems", "qty", "quantity", "units", "pcs"),
    "returns": ("returns", "return", "returned", "refunds", "refund", "salesreturn"),
}


def _find(df: pd.DataFrame, role: str, taken: set[str]) -> str | None:
    for col in df.columns:
        n = norm(col)
        if col not in taken and any(n == s or (len(s) >= 5 and n.startswith(s)) for s in EXTRA[role]):
            return col
    return None


def extract(df: pd.DataFrame, cmap: dict[str, str]) -> dict[str, list] | None:
    """{'date': [...], 'sales': [...], optional 'orders'/'items'/'returns': [...]}, one entry per date. None if unusable."""
    if "date" not in cmap or "revenue" not in cmap:
        return None
    taken = {cmap["date"], cmap["revenue"], cmap.get("product")}
    cols = {"date": to_dates(df[cmap["date"]])[0].dt.normalize(), "sales": to_number(df[cmap["revenue"]])}
    for role in EXTRA:
        col = cmap.get("units") if role == "items" and cmap.get("units") else _find(df, role, taken)
        if col is not None:
            taken.add(col)
            cols[role] = to_number(df[col])
    frame = pd.DataFrame(cols).dropna(subset=["date", "sales"])
    if frame.empty:
        return None
    by_day = frame.groupby("date").sum(min_count=1).sort_index()
    by_day = by_day[[c for c in by_day.columns if c == "sales" or by_day[c].notna().mean() > 0.8]].fillna(0)
    out = {"date": [d.date().isoformat() for d in by_day.index]}
    out.update({c: by_day[c].astype(float).tolist() for c in by_day.columns})
    return out
