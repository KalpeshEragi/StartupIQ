"""Load the downloaded statements into one table: one row per transaction, both statement formats normalised."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def indian_transactions() -> pd.DataFrame:
    """Columns: statement, holder, date, description, direction ('debit' | 'credit'), amount."""
    rows = []
    for f in sorted((RAW / "indian").glob("*/*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        sid = f"{f.parent.name}/{f.stem}"
        for t in d["transactions"]:
            if "cr_dr" in t:  # Type 2: one amount and a CR/DR flag
                direction, amount = ("credit" if t["cr_dr"] == "CR" else "debit"), t["transaction_amount"]
                date = pd.to_datetime(t["date"], format="%d/%m/%Y")
            else:  # Type 1: separate debit and credit columns
                direction = "credit" if t.get("credit") else "debit"
                amount = t.get("credit") or t.get("debit")
                date = pd.to_datetime(t["date"][:10])
            rows.append({"statement": sid, "holder": d["account_holder"], "date": date, "description": t["description"].strip(),
                         "direction": direction, "amount": float(amount or 0), "failed": bool(t.get("failed"))})
    return pd.DataFrame(rows)


def us_benchmark() -> pd.DataFrame:
    """Columns: description, direction, category. The source prefixes each description with [debit] or [credit]."""
    df = pd.read_csv(RAW / "us" / "transactions-synthetic.csv")
    sign = df.description.str.extract(r"^\[(debit|credit)\]\s*", expand=False)
    return pd.DataFrame({"description": df.description.str.replace(r"^\[(debit|credit)\]\s*", "", regex=True),
                         "direction": sign.fillna("debit"), "category": df.category})
