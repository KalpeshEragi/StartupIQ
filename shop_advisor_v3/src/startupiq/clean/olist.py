"""Olist cleaning rules and market tables (seller x category x month).

Competitor-move labels are NOT built here; they belong to Step 4 (decision 0006).
Decisions: devdocs/decisions/0010-cleaning-rules.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from startupiq.clean.log import CleaningLog

RAW = Path("data/raw/olist_hf")
WINDOW = ("2017-01", "2018-08")  # before: under 1k orders/month; after: data stops mid-October with few items
DROP_STATUS = {"canceled", "unavailable", "created"}
MIN_SELLERS_FOR_SHARE = 5


def load() -> dict[str, pd.DataFrame]:
    names = ["orders", "order_items", "products", "sellers", "order_reviews"]
    t = {n: pd.read_csv(RAW / f"olist_{n}_dataset.csv") for n in names}
    t["translation"] = pd.read_csv(RAW / "product_category_name_translation.csv")
    t["orders"]["order_purchase_timestamp"] = pd.to_datetime(t["orders"].order_purchase_timestamp)
    return t


def join_items(t: dict[str, pd.DataFrame], log: CleaningLog) -> pd.DataFrame:
    items = t["order_items"]
    d = items.merge(t["orders"][["order_id", "order_status", "order_purchase_timestamp"]], on="order_id", how="inner", validate="many_to_one")
    d = d.merge(t["products"][["product_id", "product_category_name"]], on="product_id", how="left", validate="many_to_one")
    d = d.merge(t["translation"], on="product_category_name", how="left")
    d["category"] = d.product_category_name_english.fillna(d.product_category_name).fillna("unknown")
    log.record("join_items", len(items), len(d), f"{int((d.category == 'unknown').sum()):,} items with no category labelled 'unknown'")
    return d.drop(columns=["product_category_name", "product_category_name_english"])


def drop_failed_orders(d: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    out = d[~d.order_status.isin(DROP_STATUS)]
    log.record("drop_failed_orders", len(d), len(out), f"Statuses removed: {sorted(DROP_STATUS)} (never sold)")
    return out


def keep_window(d: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    month = d.order_purchase_timestamp.dt.to_period("M")
    out = d[(month >= WINDOW[0]) & (month <= WINDOW[1])].assign(month=month)
    log.record("keep_window", len(d), len(out), f"Months {WINDOW[0]}..{WINDOW[1]}; sparse start and truncated end removed")
    return out


def clean_reviews(r: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    out = r.drop_duplicates("review_id")
    log.record("clean_reviews", len(r), len(out), "Duplicate review_id rows removed (same review attached to several orders)")
    return out


def market_tables(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scm = d.groupby(["category", "seller_id", "month"]).agg(
        revenue=("price", "sum"), items=("order_item_id", "size"), orders=("order_id", "nunique"),
        median_price=("price", "median"), products=("product_id", "nunique")).reset_index()
    cm = scm.groupby(["category", "month"]).agg(revenue=("revenue", "sum"), sellers=("seller_id", "nunique"), orders=("orders", "sum")).reset_index()
    scm = scm.merge(cm[["category", "month", "revenue", "sellers"]].rename(columns={"revenue": "category_revenue", "sellers": "category_sellers"}),
                    on=["category", "month"])
    scm["share"] = scm.revenue / scm.category_revenue
    cm = cm.merge(scm.groupby(["category", "month"]).share.apply(lambda s: float((s ** 2).sum())).rename("hhi").reset_index(), on=["category", "month"])
    scm["month"], cm["month"] = scm.month.astype(str), cm.month.astype(str)
    return scm, cm


def run(out_dir: Path = Path("data/processed")) -> CleaningLog:
    log = CleaningLog("olist")
    t = load()
    log.record("load", len(t["order_items"]), len(t["order_items"]), f"{len(t['orders']):,} orders, {t['order_items'].seller_id.nunique():,} sellers")
    d = join_items(t, log)
    d = drop_failed_orders(d, log)
    d = keep_window(d, log)
    scm, cm = market_tables(d)
    eligible = cm[cm.sellers >= MIN_SELLERS_FOR_SHARE]
    log.record("market_tables", len(scm), len(scm),
               f"Aggregation (rows are seller-category-months, built from {len(d):,} items); {len(eligible)} of {len(cm)} category-months have {MIN_SELLERS_FOR_SHARE}+ sellers (used for share models)")
    reviews = clean_reviews(t["order_reviews"], log)
    out_dir.mkdir(parents=True, exist_ok=True)
    d.assign(month=d.month.astype(str)).to_parquet(out_dir / "olist_items.parquet", index=False)
    scm.to_parquet(out_dir / "olist_seller_category_month.parquet", index=False)
    cm.to_parquet(out_dir / "olist_category_month.parquet", index=False)
    reviews.to_parquet(out_dir / "olist_reviews.parquet", index=False)
    t["sellers"].to_parquet(out_dir / "olist_sellers.parquet", index=False)
    log.save(out_dir)
    return log
