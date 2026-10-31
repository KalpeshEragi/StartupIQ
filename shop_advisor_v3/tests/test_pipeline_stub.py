import numpy as np
import pandas as pd

from pathlib import Path

from startupiq.contracts import BusinessContext, Source, Tier
from startupiq.intake import read_sales, tier_for
from startupiq.pipeline import analyze, analyze_file

PACK = Path(__file__).parent / "fixtures" / "intake" / "files"


def daily_sales(days: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    return pd.DataFrame({"Order Date": dates.strftime("%d/%m/%Y"), "Total Sales (INR)": rng.integers(500, 1500, days)})


def test_analyze_file_refused_shows_reason_only():
    report = analyze_file(PACK / "10_no_date.csv")
    assert report.tier == Tier.REFUSED and [c.key for c in report.cards] == ["intake"]
    assert "date column" in report.cards[0].headline


def test_analyze_file_monthly_upload():
    report = analyze_file(PACK / "23_monthly_totals.csv", BusinessContext(cash_on_hand=300000, monthly_fixed_costs=150000, gross_margin=0.4))
    assert report.cards[0].headline == "20 months read"
    assert {"growth_recent", "runway_months", "cash_risk"} <= {f.name for f in report.facts}  # inputs plus rule conclusions


def test_questions_reach_the_intake_card():
    report = analyze_file(PACK / "17_gap_in_middle.csv")
    assert "gaps" in report.cards[0].details["needs_confirmation"]


def test_tiers_follow_weeks():
    assert tier_for(4) == Tier.SUMMARY_ONLY
    assert tier_for(10) == Tier.BASIC
    assert tier_for(30) == Tier.STANDARD
    assert tier_for(60) == Tier.FULL


def test_refuses_without_date_column():
    got = read_sales(pd.DataFrame({"foo": [1, 2], "sales": [3, 4]}))
    assert got.tier == Tier.REFUSED
    assert "date" in got.messages[0]


def test_refuses_too_few_rows():
    assert read_sales(daily_sales(5)).tier == Tier.REFUSED


def test_full_year_gets_forecast_and_all_cards():
    report = analyze(daily_sales(400), BusinessContext(cash_on_hand=200000, monthly_fixed_costs=50000, gross_margin=0.3))
    keys = [c.key for c in report.cards if c.key not in ("segment", "category")]
    assert report.tier == Tier.FULL
    assert keys == ["intake", "forecast", "money", "warnings", "market", "rivals", "next_move"]
    fc = next(c for c in report.cards if c.key == "forecast")
    assert fc.source == Source.PREDICTED and fc.details["model"].startswith("auto: ") and len(fc.details["p50"]) == 12
    assert all(lo <= mid <= hi for lo, mid, hi in zip(fc.details["p10"], fc.details["p50"], fc.details["p90"]))
    assert any(f.name == "runway_months" for f in report.facts)


def test_short_history_locks_forecast():
    report = analyze(daily_sales(30))
    assert report.tier == Tier.SUMMARY_ONLY
    forecast = next(c for c in report.cards if c.key == "forecast")
    assert forecast.locked_reason


def test_monthly_upload_forecasts_three_months():
    report = analyze_file(PACK / "23_monthly_totals.csv")
    fc = next(c for c in report.cards if c.key == "forecast")
    assert len(fc.details["p50"]) == 3 and "months" in fc.headline


def test_rule_cards_filled_and_unsafe_moves_blocked():
    ctx = BusinessContext(cash_on_hand=40000, monthly_fixed_costs=25000, gross_margin=0.4, stock_value=5000,
                          supplier_lead_time_days=10, competitors=["Sharma Snacks"])
    report = analyze_file(PACK / "04_title_rows.xlsx", ctx)
    cards = {c.key: c for c in report.cards}
    assert all(cards[k].locked_reason is None for k in ("warnings", "market", "rivals", "next_move"))
    assert "runway_critical" in cards["warnings"].details["warnings"]
    assert cards["rivals"].source == Source.INFERRED
    assert report.recommendations and all(r.rules_fired for r in report.recommendations)
    assert "price_coordination" not in [r.action for r in report.recommendations]
