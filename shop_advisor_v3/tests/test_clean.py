import pandas as pd
import pytest

from startupiq.clean import olist, rossmann, uci_retail
from startupiq.clean.log import CleaningLog


def ross_frame() -> pd.DataFrame:
    dates = pd.date_range("2014-06-29", "2015-01-03")
    rows = []
    for store in (1, 2):
        for d in dates:
            if store == 2 and rossmann.GAP_START <= d <= rossmann.GAP_END:
                continue  # store 2 has the known 2014 gap
            rows.append({"Store": store, "Date": d, "Sales": 100.0, "Open": 1, "Customers": 10, "Promo": 0,
                         "StateHoliday": "0", "SchoolHoliday": 0, "DayOfWeek": d.dayofweek + 1})
    return pd.DataFrame(rows)


def test_zero_sales_on_open_day_becomes_missing_not_dropped():
    df = ross_frame()
    df.loc[0, "Sales"] = 0
    log = CleaningLog("t")
    out = rossmann.mark_zero_sales_on_open_days(df, log)
    assert len(out) == len(df) and pd.isna(out.loc[0, "Sales"]) and out.loc[0, "sales_missing"]
    assert log.entries[0].changed == 0


def test_known_gap_flags_only_gap_store():
    out = rossmann.flag_known_gap(ross_frame(), CleaningLog("t"))
    assert out.groupby("Store").store_has_2014_gap.first().to_dict() == {1: False, 2: True}


def test_outlier_is_flagged_not_dropped():
    df = ross_frame()
    df["Sales"] = 100.0 + (df.index % 7)  # small natural variation so MAD > 0
    df.loc[5, "Sales"] = 100000
    out = rossmann.flag_outliers(df, CleaningLog("t"))
    assert len(out) == len(df) and out.loc[5, "outlier"] and out.outlier.sum() == 1


def uci_frame() -> pd.DataFrame:
    t = pd.Timestamp("2010-01-04")
    return pd.DataFrame([
        {"Invoice": "1", "StockCode": "85123A", "Quantity": 6, "Price": 2.5, "CustomerID": 1.0, "InvoiceDate": t},
        {"Invoice": "2", "StockCode": "85123A", "Quantity": 6, "Price": 2.5, "CustomerID": 1.0, "InvoiceDate": t + pd.Timedelta(days=1)},
        {"Invoice": "C3", "StockCode": "85123A", "Quantity": -6, "Price": 2.5, "CustomerID": 1.0, "InvoiceDate": t + pd.Timedelta(days=2)},
        {"Invoice": "C4", "StockCode": "22423", "Quantity": -1, "Price": 9.0, "CustomerID": 2.0, "InvoiceDate": t},  # partial return, no match
        {"Invoice": "5", "StockCode": "POST", "Quantity": 1, "Price": 18.0, "CustomerID": 1.0, "InvoiceDate": t},
        {"Invoice": "6", "StockCode": "22423", "Quantity": 0, "Price": 0.0, "CustomerID": None, "InvoiceDate": t},
    ])


def test_non_products_removed():
    out = uci_retail.drop_non_products(uci_frame(), CleaningLog("t"))
    assert "POST" not in set(out.StockCode)


def test_cancellation_removes_latest_matching_sale_only():
    kept, returns = uci_retail.net_off_cancellations(uci_frame(), CleaningLog("t"))
    assert list(kept.Invoice) == ["1", "5", "6"]  # sale "2" is the latest match for C3 and goes; "1" stays
    assert set(returns.Invoice) == {"C3", "C4"}


def test_non_positive_lines_dropped():
    out = uci_retail.drop_non_positive(uci_frame(), CleaningLog("t"))
    assert (out.Quantity > 0).all() and (out.Price > 0).all()


def olist_items() -> pd.DataFrame:
    return pd.DataFrame({
        "category": ["toys"] * 4, "seller_id": ["a", "a", "b", "c"], "month": pd.PeriodIndex(["2017-03"] * 4, freq="M"),
        "price": [10.0, 30.0, 40.0, 20.0], "order_item_id": [1, 1, 1, 1], "order_id": ["o1", "o2", "o3", "o4"],
        "product_id": ["p1", "p2", "p3", "p4"], "order_status": ["delivered"] * 4,
    })


def test_market_shares_sum_to_one_and_hhi():
    scm, cm = olist.market_tables(olist_items())
    assert scm.share.sum() == pytest.approx(1.0)
    assert scm.set_index("seller_id").share.to_dict() == pytest.approx({"a": 0.4, "b": 0.4, "c": 0.2})
    assert cm.hhi.iloc[0] == pytest.approx(0.4 ** 2 + 0.4 ** 2 + 0.2 ** 2)


def test_failed_orders_dropped():
    d = olist_items()
    d.loc[0, "order_status"] = "canceled"
    out = olist.drop_failed_orders(d, CleaningLog("t"))
    assert len(out) == 3


def test_log_markdown_has_counts():
    log = CleaningLog("demo")
    log.record("rule_x", 1000, 900, "why")
    md = log.to_markdown()
    assert "| rule_x | 1,000 | 900 | 100 | why |" in md
