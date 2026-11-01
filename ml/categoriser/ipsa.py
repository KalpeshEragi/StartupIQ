"""The labelled training data: IPSA (UK MP office) expense claims mapped onto startup expense categories.

    python -m categoriser.ipsa          writes data/interim/ipsa_labelled.parquet and prints the split sizes

Splits are by financial year, the way the product meets data: models learn from the past and are judged on later
years, with vendors and phrasing they may never have seen.
    train  2010-11 .. 2021-22
    val    2022-23 .. 2023-24     (model and threshold choice)
    test   2024-25 .. 2026-27     (scored once)
"""
from __future__ import annotations

import pandas as pd

from categoriser.data import RAW, ROOT
from categoriser.taxonomy import ipsa_category
from categoriser.text import clean

INTERIM = ROOT / "data" / "interim"
VAL_YEARS = {"22_23", "23_24"}
TEST_YEARS = {"24_25", "25_26", "26_27"}


def split_of(year: str) -> str:
    return "test" if year in TEST_YEARS else ("val" if year in VAL_YEARS else "train")


def build() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "ipsa" / "mp_expenses.parquet",
                         columns=["year", "date", "cost_type", "short_description", "details", "amount_claimed"])
    df["category"] = df.cost_type.map(ipsa_category)
    raw_text = (df.short_description.fillna("") + " " + df.details.fillna("")).str.strip()
    df["text"] = raw_text.map(clean)
    keep = df.category.notna() & (df.text.str.len() >= 3) & (df.amount_claimed > 0)
    df = df[keep].drop_duplicates(["year", "date", "cost_type", "text", "amount_claimed"])
    out = pd.DataFrame({"text": df.text, "raw": raw_text[df.index], "amount": df.amount_claimed.astype(float),
                        "category": df.category, "cost_type": df.cost_type, "year": df.year,
                        "split": df.year.map(split_of)})
    return out.reset_index(drop=True)


def load() -> pd.DataFrame:
    path = INTERIM / "ipsa_labelled.parquet"
    if not path.exists():
        INTERIM.mkdir(parents=True, exist_ok=True)
        build().to_parquet(path)
    return pd.read_parquet(path)


if __name__ == "__main__":
    data = load()
    print(f"{len(data):,} labelled rows, {data.text.nunique():,} distinct cleaned texts")
    print(pd.crosstab(data.category, data.split).loc[:, ["train", "val", "test"]].to_string())
    for split in ("val", "test"):
        seen = set(data.text[data.split == "train"])
        part = data[data.split == split]
        print(f"{split}: {part.text.isin(seen).mean():.1%} of rows have a text already seen in training")
