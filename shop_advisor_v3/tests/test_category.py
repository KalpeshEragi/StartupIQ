"""Business-type classifier: synthetic shops, features from any register, and intake picking up the extra columns."""

import numpy as np
import pandas as pd
import pytest

from startupiq import category, synthetic
from startupiq.intake import read_table


def _shop(cat="Restaurant & cafe", weeks=60, seed=0):
    df, meta = synthetic.business(np.random.default_rng(seed), cat, pd.Timestamp("2026-06-28"), weeks)
    return df, meta


def test_synthetic_shop_follows_its_type():
    df, meta = _shop("Grocery & kirana")
    assert len(df) == 60 * 7 and (df["items"] >= df.orders).all() and (df.sales > 0).all()
    assert 3 < df["items"].sum() / df.orders.sum() < 20  # a kirana bill has many items


def test_features_work_with_sales_only_and_weekly_rows():
    df, _ = _shop()
    full = category.features(df)
    only_sales = category.features(df[["date", "sales"]])
    weekly = category.features(df.set_index("date").resample("W-SUN").sum().reset_index())
    assert not np.isnan(full["log_avg_bill"]) and np.isnan(only_sales["log_avg_bill"])
    assert not np.isnan(full["weekend_share"]) and np.isnan(weekly["weekend_share"])
    assert full["weekend_share"] > 2 / 7  # restaurants are busy at weekends


def test_intake_reads_bills_items_and_returns():
    df, _ = _shop("Clothing & fashion", weeks=30)
    upload = pd.DataFrame({"Date": df.date.dt.strftime("%d/%m/%Y"), "Sales (₹)": df.sales, "No. of bills": df.orders,
                           "Items sold": df["items"], "Returns (₹)": df.returns})
    got = read_table(upload)
    assert set(got.signals) == {"date", "sales", "orders", "items", "returns"}
    assert got.column_map["revenue"] == "Sales (₹)"
    assert sum(got.signals["sales"]) == pytest.approx(df.sales.sum())


@pytest.mark.skipif(category.load() is None, reason="run python -m startupiq.category first")
def test_classifier_recognises_clear_shops():
    for cat in ("Stationery & books", "Electronics & mobiles"):
        df, _ = _shop(cat, weeks=80, seed=3)
        got = category.classify(df)
        assert got["best_guess"] == cat and len(got["probabilities"]) == 6
        assert abs(sum(p["p"] for p in got["probabilities"]) - 1) < 0.01
    assert category.classify(df.head(5)) is None
