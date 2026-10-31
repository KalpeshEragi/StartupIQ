"""Leakage and label tests for Step 4 features."""

import numpy as np
import pandas as pd
import pytest

from startupiq.evaluate.metrics import mase, mase_scale, rmspe
from startupiq.features import olist as fo
from startupiq.features import rossmann as fr
from startupiq.features import weekly as fw
from startupiq.features.splits import olist_split


def ross_daily(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for store in (1, 2):
        for d in pd.date_range("2013-01-01", "2014-12-31"):
            open_ = int(d.dayofweek != 6)
            rows.append({"Store": store, "Date": d, "DayOfWeek": d.dayofweek + 1, "Open": open_,
                         "Sales": float(rng.integers(4000, 8000)) * open_, "Customers": 500 * open_,
                         "Promo": int(d.day % 14 < 5), "StateHoliday": "a" if (d.month, d.day) == (12, 25) else "0",
                         "SchoolHoliday": 0, "StoreType": "a", "Assortment": "a", "CompetitionDistance": 1000.0,
                         "CompetitionOpenSinceMonth": 1.0, "CompetitionOpenSinceYear": 2010.0, "Promo2": 0,
                         "Promo2SinceWeek": 0.0, "Promo2SinceYear": 0.0, "PromoInterval": "none"})
    return pd.DataFrame(rows)


def test_rossmann_features_ignore_sales_after_origin():
    origin = pd.Timestamp("2014-11-01")
    a = ross_daily()
    b = a.copy()
    b.loc[b.Date >= origin, "Sales"] *= 10  # change only the future
    fa = fr.build(a, origin).set_index(["Store", "Date"])[fr.NUMERIC]
    fb = fr.build(b, origin).set_index(["Store", "Date"])[fr.NUMERIC]
    pd.testing.assert_frame_equal(fa, fb)


def test_rossmann_never_uses_customers():
    assert "Customers" not in fr.FEATURES


def test_weekly_features_ignore_values_from_origin_on():
    y = np.arange(120, dtype=float)
    weeks = pd.date_range("2022-01-02", periods=120, freq="W-SUN")
    f1 = fw.origin_features(y, 90, weeks, 0)
    y2 = y.copy(); y2[90:] = 1e6
    f2 = fw.origin_features(y2, 90, weeks, 0)
    pd.testing.assert_frame_equal(f1, f2)
    assert f1.seasonal_same_week.iloc[0] == pytest.approx(y[90 - 52] / (y[78:90].mean() + 1))


def olist_items(rows) -> pd.DataFrame:
    out = []
    for i, (seller, month, product, price, n) in enumerate(rows):
        for k in range(n):
            out.append({"order_id": f"o{i}_{k}", "order_item_id": 1, "product_id": product, "seller_id": seller,
                        "price": price, "category": "toys", "month": month})
    return pd.DataFrame(out)


def test_olist_labels():
    months = [str(m) for m in pd.period_range("2017-01", "2017-12", freq="M")]
    rows = []
    for m in months:
        rows.append(("cutter", m, "p1", 100.0 if m < "2017-06" else 80.0, 4))      # same product 20% cheaper from June
        rows.append(("mixer", m, "p2" if m < "2017-06" else "p3", 100.0 if m < "2017-06" else 50.0, 4))  # different product, not a cut
        if m <= "2017-05":
            rows.append(("leaver", m, "p4", 30.0, 3))                                 # stops after May
        rows.append(("grower", m, "p5", 20.0, 2 if m != "2017-06" else 8))            # jumps in June
    d = fo.build(olist_items(rows), pd.DataFrame({"seller_id": ["cutter", "mixer", "leaver", "grower"], "seller_state": "SP"}),
                 pd.DataFrame({"order_id": [], "review_score": []}))
    at = d.set_index(["seller_id", d.month.astype(str)]).label
    assert at[("cutter", "2017-05")] == "price_cut"
    assert at[("mixer", "2017-05")] != "price_cut"
    assert at[("leaver", "2017-05")] == "exit"
    assert at[("grower", "2017-05")] == "expansion"
    assert at[("cutter", "2017-07")] == "stagnate"
    # Rows whose 3-month label window runs past the data are dropped.
    assert d.month.max() <= pd.Period("2017-09", "M")


def test_olist_split_boundaries():
    m = pd.Series(pd.PeriodIndex(["2017-02", "2017-03", "2017-11", "2017-12", "2018-02", "2018-03", "2018-05", "2018-06"], freq="M"))
    assert olist_split(m).where(lambda s: s.notna(), None).tolist() == [None, "train", "train", "val", "val", "test", "test", None]


def test_metrics():
    assert rmspe([100, 200, 0], [110, 180, 5]) == pytest.approx(np.sqrt((0.01 + 0.01) / 2))
    assert mase_scale([1, 2, 3, 4], 1) == 1.0
    assert mase([5, 5], [4, 6], 1.0) == 1.0


def test_competitor_decision_rule_prefers_calibration_on_ties():
    from startupiq.models.competitor import _choose
    results = [
        {"model": "Majority", "macro_f1": [0.2], "brier": [float("nan")]},
        {"model": "A", "macro_f1": [0.380, 0.383, 0.377], "brier": [0.60, 0.60, 0.60]},
        {"model": "B", "macro_f1": [0.379, 0.378, 0.380], "brier": [0.55, 0.55, 0.55]},
        {"model": "C", "macro_f1": [0.30, 0.31, 0.29], "brier": [0.40, 0.40, 0.40]},
    ]
    assert _choose(results) == "B"  # A is best, B is within one sd of A and better calibrated; C is not tied


def test_regime_labels():
    from startupiq.models.regime import _label
    assert [_label(x) for x in (0.2, 0.0, -0.2, 0.05)] == ["rising", "flat", "falling", "flat"]
