"""Olist seller panel: competitor-move labels (decision 0006) and features known at month t.

One row = one seller in one category in month t, for sellers active in t with at least
one earlier active month. Features use months <= t (reviews: <= t-1, because a review
arrives after delivery). Labels look at t+1..t+3 only, so a row's label never reaches
past its split's end (see features/splits.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from startupiq.features.splits import olist_split

PRICE_CUT = 0.95
EXPANSION = 1.3
EXPANSION_MIN_ORDERS = 3
EXIT_MONTHS = 3
LABELS = ["exit", "price_cut", "expansion", "stagnate"]

NUMERIC = ["orders", "orders_lag1", "orders_lag2", "log_revenue", "revenue_growth", "items", "products", "log_median_price",
           "price_position", "price_change_prev", "share", "share_lag1", "share_change", "rank_pct", "category_sellers",
           "category_hhi", "category_growth", "active_months", "months_since_first", "seller_categories",
           "review_mean", "review_count", "month_of_year"]
CATEGORICAL = ["category", "seller_state"]
FEATURES = NUMERIC + CATEGORICAL


def _same_product_price_ratio(items: pd.DataFrame) -> pd.DataFrame:
    """Median over products sold in both months of price(m+1)/price(m), per (category, seller, m)."""
    p = items.groupby(["category", "seller_id", "product_id", "month"]).price.median().reset_index()
    nxt = p.assign(month=p.month - 1).rename(columns={"price": "price_next"})
    both = p.merge(nxt, on=["category", "seller_id", "product_id", "month"])
    both["ratio"] = both.price_next / both.price
    return both.groupby(["category", "seller_id", "month"]).ratio.median().rename("price_ratio_next").reset_index()


def build(items: pd.DataFrame | None = None, sellers: pd.DataFrame | None = None, reviews: pd.DataFrame | None = None) -> pd.DataFrame:
    """Defaults read data/processed; tests pass small frames instead."""
    items = (pd.read_parquet("data/processed/olist_items.parquet") if items is None else items).copy()
    items["month"] = pd.PeriodIndex(items.month, freq="M")
    sellers = (pd.read_parquet("data/processed/olist_sellers.parquet") if sellers is None else sellers)[["seller_id", "seller_state"]]
    reviews = (pd.read_parquet("data/processed/olist_reviews.parquet") if reviews is None else reviews)[["order_id", "review_score"]]

    agg = items.groupby(["category", "seller_id", "month"]).agg(
        orders=("order_id", "nunique"), items=("order_item_id", "size"), revenue=("price", "sum"),
        median_price=("price", "median"), products=("product_id", "nunique")).reset_index()
    months = pd.period_range(items.month.min(), items.month.max(), freq="M")
    pairs = agg[["category", "seller_id"]].drop_duplicates()
    grid = pairs.merge(pd.DataFrame({"month": months}), how="cross")
    d = grid.merge(agg, on=["category", "seller_id", "month"], how="left")
    d[["orders", "items", "revenue", "products"]] = d[["orders", "items", "revenue", "products"]].fillna(0)
    d = d.sort_values(["category", "seller_id", "month"]).reset_index(drop=True)
    g = d.groupby(["category", "seller_id"], sort=False)

    cat = d.groupby(["category", "month"]).agg(cat_rev=("revenue", "sum"), category_sellers=("orders", lambda s: int((s > 0).sum()))).reset_index()
    cat["category_growth"] = cat.groupby("category").cat_rev.transform(lambda s: s / s.shift(1).replace(0, np.nan))
    cat_med = items.groupby(["category", "month"]).price.median().rename("cat_median_price").reset_index()
    d = d.merge(cat, on=["category", "month"], how="left").merge(cat_med, on=["category", "month"], how="left")
    d["share"] = d.revenue / d.cat_rev.replace(0, np.nan)
    d["category_hhi"] = d.groupby(["category", "month"]).share.transform(lambda s: float((s.fillna(0) ** 2).sum()))
    d["rank_pct"] = d.groupby(["category", "month"]).revenue.rank(pct=True)
    g = d.groupby(["category", "seller_id"], sort=False)

    d["orders_lag1"], d["orders_lag2"] = g.orders.shift(1), g.orders.shift(2)
    d["share_lag1"] = g.share.shift(1)
    d["share_change"] = d.share - d.share_lag1
    d["log_revenue"] = np.log1p(d.revenue)
    d["revenue_growth"] = d.revenue / g.revenue.shift(1).replace(0, np.nan)
    d["log_median_price"] = np.log1p(d.median_price)
    d["price_position"] = d.median_price / d.cat_median_price
    active = (d.orders > 0).astype(int)
    d["active_months"] = active.groupby([d.category, d.seller_id]).cumsum()
    first = d.month.where(d.orders > 0).groupby([d.category, d.seller_id]).transform("min")
    d["months_since_first"] = (d.month - first).apply(lambda x: x.n if pd.notna(x) else np.nan)
    sc = items.groupby(["seller_id", "month"]).category.nunique().rename("cats").reset_index().sort_values("month")
    d = d.merge(sc, on=["seller_id", "month"], how="left")
    d["seller_categories"] = d.cats.fillna(0)

    ratio = _same_product_price_ratio(items)
    d = d.merge(ratio, on=["category", "seller_id", "month"], how="left")
    d["price_change_prev"] = d.groupby(["category", "seller_id"]).price_ratio_next.shift(1)

    # Reviews of orders bought up to t-1 (a review is written after delivery).
    rv = items[["order_id", "seller_id", "category", "month"]].drop_duplicates().merge(reviews, on="order_id")
    rv = rv.groupby(["category", "seller_id", "month"]).review_score.agg(["sum", "count"]).reset_index()
    d = d.merge(rv, on=["category", "seller_id", "month"], how="left").fillna({"sum": 0, "count": 0})
    g = d.groupby(["category", "seller_id"], sort=False)
    cum_sum, cum_cnt = g["sum"].cumsum() - d["sum"], g["count"].cumsum() - d["count"]
    d["review_mean"] = cum_sum / cum_cnt.replace(0, np.nan)
    d["review_count"] = cum_cnt
    d["month_of_year"] = d.month.dt.month
    d = d.merge(sellers, on="seller_id", how="left")

    # Labels from t+1..t+3.
    nxt = [g.orders.shift(-k) for k in range(1, EXIT_MONTHS + 1)]
    known = pd.concat(nxt, axis=1).notna().all(axis=1)
    exit_ = pd.concat(nxt, axis=1).eq(0).all(axis=1)
    price_cut = d.price_ratio_next < PRICE_CUT
    expansion = (nxt[0] > EXPANSION * d.orders) & (nxt[0] >= EXPANSION_MIN_ORDERS)
    d["label"] = np.select([exit_, price_cut, expansion], ["exit", "price_cut", "expansion"], "stagnate")
    d.loc[~known, "label"] = None
    d["label_prev"] = d.groupby(["category", "seller_id"], sort=False).label.shift(1).fillna("stagnate")
    d["share_next"] = g.share.shift(-1).fillna(0).where(d.month < d.month.max())  # NaN only when t+1 is past the data

    rows = d[(d.orders > 0) & (d.active_months >= 2) & d.label.notna()].copy()
    rows["split"] = olist_split(rows.month)
    rows = rows[rows.split.notna()]
    for c in CATEGORICAL:
        rows[c] = rows[c].astype(str).astype("category")
    return rows.reset_index(drop=True)
