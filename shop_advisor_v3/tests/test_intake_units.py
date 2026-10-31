from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from startupiq.intake import read_table
from startupiq.intake.columns import assign_roles, name_score
from startupiq.intake.parsing import looks_like_period, to_dates, to_number
from startupiq.intake.reader import find_header_row
from startupiq.intake.series import granularity_of, tier_for

TODAY = date(2026, 10, 24)
RAW = Path("data/raw")


def test_money_formats():
    got = to_number(pd.Series(["₹1,25,000.50", "Rs. 450/-", "(300)", " 12 ", "-", "abc", "£9.99"]))
    assert got.tolist()[:4] == [125000.5, 450.0, -300.0, 12.0]
    assert got.iloc[4] != got.iloc[4] and got.iloc[5] != got.iloc[5]  # NaN
    assert got.iloc[6] == pytest.approx(9.99)


@pytest.mark.parametrize("values,how,month", [
    (["2024-03-01", "2024-03-02"], "iso", 3),
    (["13/03/2024", "01/03/2024"], "day_first", 3),
    (["03/13/2024", "03/01/2024"], "month_first", 3),
    (["01/03/2024", "02/03/2024"], "day_first_ambiguous", 3),
    (["Jan-2024", "Feb-2024"], "month_only", 1),
    ([45352, 45353], "excel_serial", 3),  # 45352 = 2024-03-01
])
def test_date_rules(values, how, month):
    parsed, rule = to_dates(pd.Series(values))
    assert rule == how and parsed.iloc[0].month == month


def test_period_headers():
    assert looks_like_period("Jan-2024") and looks_like_period("2024-03") and looks_like_period("March 2024")
    assert not looks_like_period("Product") and not looks_like_period("Qty") and not looks_like_period("2024")


def test_header_row_after_title_rows():
    grid = pd.DataFrame([["Shop report", None, None], [None, None, None], ["Date", "Item", "Amount"],
                         ["01/01/2024", "Tea", "100"], ["02/01/2024", "Tea", "120"]])
    assert find_header_row(grid.dropna(how="all").reset_index(drop=True)) == 1


def test_headerless_file_detected():
    grid = pd.DataFrame([["01/01/2024", "100"], ["02/01/2024", "120"]])
    assert find_header_row(grid) is None


def test_name_penalties_and_preferences():
    assert name_score("Invoice No", "revenue") < 0.5
    assert name_score("freight_value", "revenue") < 0.5
    assert name_score("Net Amount", "revenue") >= 0.7
    assert name_score("order_purchase_timestamp", "date") > name_score("shipping_limit_date", "date")


def test_content_beats_misleading_name():
    df = pd.DataFrame({"Date": ["not a date"] * 30, "When": pd.date_range("2024-01-01", periods=30).astype(str),
                       "Sales": range(30)})
    assert assign_roles(df).column_map["date"] == "When"


def test_id_and_year_columns_never_revenue():
    df = pd.DataFrame({"Column1": pd.date_range("2024-01-01", periods=40).astype(str),
                       "Column2": range(1000, 1040), "Column3": [2024] * 40, "Column4": [float(i % 7) * 100 + 50 for i in range(40)]})
    assert assign_roles(df).column_map["revenue"] == "Column4"


def test_granularity():
    d = pd.Series(pd.date_range("2024-01-01", periods=60, freq="D"))
    assert granularity_of(d) == "D"
    assert granularity_of(pd.concat([d, d, d])) == "transactions"
    assert granularity_of(pd.Series(pd.date_range("2024-01-01", periods=20, freq="W"))) == "W"
    assert granularity_of(pd.Series(pd.date_range("2024-01-01", periods=20, freq="MS"))) == "M"


def test_tier_thresholds():
    assert [tier_for(w).value for w in (4, 8, 26, 52)] == ["summary_only", "basic", "standard", "full"]


def test_user_override_wins():
    df = pd.DataFrame({"a": pd.date_range("2024-01-01", periods=120).astype(str), "b": range(120), "c": range(120, 240)})
    got = read_table(df, {"date": "a", "revenue": "c"}, today=TODAY)
    assert got.column_map["revenue"] == "c" and got.confidence["revenue"] == 1.0 and "revenue" not in got.needs_confirmation


def test_future_rows_removed():
    df = pd.DataFrame({"date": pd.date_range("2026-06-01", periods=200).astype(str), "sales": 100.0})
    got = read_table(df, today=TODAY)
    assert any("future-dated" in m for m in got.cleaning_log)
    assert max(got.series[0].dates) <= TODAY


@pytest.mark.skipif(not (RAW / "olist_hf").exists(), reason="Olist raw data not downloaded")
def test_real_olist_export_uses_purchase_date_not_shipping_or_freight():
    items = pd.read_csv(RAW / "olist_hf/olist_order_items_dataset.csv").merge(
        pd.read_csv(RAW / "olist_hf/olist_orders_dataset.csv"), on="order_id")
    seller = items.seller_id.value_counts().index[0]
    got = read_table(items[items.seller_id == seller], today=TODAY)
    assert got.column_map["date"] == "order_purchase_timestamp"
    assert "revenue" not in got.column_map and got.column_map["price"] == "price"
    assert got.tier.value == "full"


@pytest.mark.skipif(not (RAW / "rossmann_hf").exists(), reason="Rossmann raw data not downloaded")
def test_real_rossmann_store():
    df = pd.read_csv(RAW / "rossmann_hf/train.csv", low_memory=False, nrows=20000)
    got = read_table(df[df.Store == 1], today=TODAY)
    assert got.column_map["date"] == "Date" and got.column_map["revenue"] == "Sales"
