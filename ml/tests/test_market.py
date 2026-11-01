from datetime import date

import pandas as pd

from market import index, pageviews
from market.sectors import SECTORS, START, articles


def test_window_starts_in_2023():
    assert START == "2023-01-01"


def test_every_sector_has_enough_articles():
    for key, spec in SECTORS.items():
        assert spec["label"]
        assert len(spec["en.wikipedia"]) >= 6, key
    assert len(articles()) == len(set(articles()))


def test_url_encodes_titles():
    u = pageviews.url("en.wikipedia", "Paper Boat (drink)", date(2023, 1, 1), date(2023, 1, 31))
    assert u.endswith("/en.wikipedia/all-access/user/Paper_Boat_%28drink%29/daily/2023010100/2023013100")
    hi = pageviews.url("hi.wikipedia", "योग", date(2023, 1, 1), date(2023, 1, 2))
    assert "%E0%A4%AF" in hi  # Hindi title is percent-encoded


def test_scale_uses_median_and_caps_spikes():
    days = pd.date_range("2023-01-01", periods=730, freq="D")
    views = pd.Series(100.0, index=days)
    views.iloc[400:] = 150.0
    views.iloc[500] = 100000.0  # one viral day
    s = index.scale(views)
    assert s.iloc[0] == 1.0 and s.iloc[450] == 1.5
    assert s.iloc[500] < 10  # capped, not 1000x


def test_scale_ignores_days_before_article_existed():
    days = pd.date_range("2023-01-01", periods=400, freq="D")
    views = pd.Series([float("nan")] * 300 + [50.0] * 100, index=days)
    s = index.scale(views)
    assert s.iloc[:300].isna().all() and s.iloc[-1] == 1.0


def test_scale_drops_tiny_articles():
    days = pd.date_range("2023-01-01", periods=400, freq="D")
    assert index.scale(pd.Series(5.0, index=days)) is None


def test_sector_index_is_median_so_one_article_cannot_move_it():
    days = pd.date_range("2023-01-01", periods=3, freq="D")
    wide = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [1.0, 1.0, 1.0], "c": [1.0, 9.0, 1.0]}, index=days)
    assert list(index.sector_index(wide)) == [100.0, 100.0, 100.0]


def test_weekly_drops_partial_weeks():
    days = pd.date_range("2023-01-04", "2023-01-22", freq="D")  # Wed .. Sun: first week partial
    w = index.weekly(pd.Series(range(len(days)), index=days, dtype="float64"))
    assert list(w.index.strftime("%Y-%m-%d")) == ["2023-01-15", "2023-01-22"]
    assert w.iloc[0] == 8.0  # mean of days 9..15 -> values 5..11


def test_merge_adds_views_across_redirect_titles():
    old = [{"timestamp": "2025100100", "views": 900}, {"timestamp": "2025100200", "views": 10}]
    new = [{"timestamp": "2025100100", "views": 5}, {"timestamp": "2025100200", "views": 3000}]
    assert pageviews.merge([old, new]) == {"2025-10-01": 905, "2025-10-02": 3010}


def test_share_removes_project_wide_decline():
    days = pd.date_range("2023-01-01", periods=400, freq="D")
    totals = pd.Series(1_000_000.0, index=days)
    totals.iloc[200:] = 500_000.0           # Wikipedia loses half its readers
    views = pd.Series(100.0, index=days)
    views.iloc[200:] = 50.0                 # the article falls by exactly the same amount
    s = index.scale(views, totals)
    assert (s.round(6) == 1.0).all()        # its share of attention did not change


def test_relative_removes_drift_shared_by_all_sectors():
    days = pd.date_range("2023-01-01", periods=2, freq="D")
    wide = pd.DataFrame({"a": [100.0, 50.0], "b": [100.0, 50.0], "ai": [100.0, 100.0]}, index=days)
    r = index.relative(wide)
    assert list(r["a"]) == [100.0, 100.0]   # fell with everyone: flat
    assert list(r["ai"]) == [100.0, 200.0]  # held up while others halved: doubled relative


def test_split_is_by_time_and_test_is_last_12_weeks():
    from market import evaluate
    w = pd.DataFrame({"a": range(100)}, index=pd.date_range("2023-01-01", periods=100, freq="W-SUN"), dtype="float64")
    p = evaluate.split(w)
    assert len(p["test"]) == 12 and len(p["val"]) == 12 and len(p["train"]) == 76
    assert p["train"].index.max() < p["val"].index.min() < p["test"].index.min()


def test_baselines_shapes():
    import numpy as np

    from market import evaluate
    hist = np.arange(120, dtype=float)
    assert list(evaluate.naive(hist)) == [119.0] * 12
    assert list(evaluate.seasonal_naive(hist)) == list(np.arange(68, 80, dtype=float))


def test_models_output_ordered_quantiles():
    import pytest
    torch = pytest.importorskip("torch")
    from market.dataset import H, L
    from market.models import ARCHS
    for arch in ARCHS.values():
        out = arch(19)(torch.randn(3, L, 6), torch.randn(3, H, 4), torch.zeros(3, dtype=torch.long))
        assert out.shape == (3, H, 3)
        assert (out[..., 0] <= out[..., 1]).all() and (out[..., 1] <= out[..., 2]).all()
