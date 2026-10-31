"""Turn intake, vitals, the forecast and Step 5 model outputs into facts for the rule engine.

Every fact has a name from FACTS, a value, a certainty (0..1) and a source:
measured (arithmetic on the founder's data), predicted (a validated model),
inferred (an estimate from proxies or base rates). Certainties come from measured
model quality where we have it (decision 0013).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from startupiq.contracts import BusinessContext, Fact, Forecast, IntakeResult, Source

WEEKS_PER_MONTH = 52 / 12
PEAK_RATIO = 1.25
TREND_THRESHOLD = 0.05
MARKET_PULSE_CF = 0.6  # "same weeks last year" rule: macro-F1 0.62 on test (decision 0012)
RECENT_SLOPE_CF = 0.45  # recent-slope rule: macro-F1 0.48 on test
RIVAL_BASE_RATES = {"exit": 0.16, "expansion": 0.19, "price_cut": 0.02}  # Olist training split
RIVAL_BASE_CF = 0.3  # a base rate says little about a particular rival
RIVAL_MODEL_CF = 0.55  # competitor random forest: macro-F1 0.37 on test, Brier 0.56
SHARE_ALERT_PRECISION = 0.69  # share drops of 2+ points predicted on the test split that really happened

FACTS = {
    "history_weeks": "weeks of usable history (M0 intake)",
    "data_tier": "intake tier (M0)",
    "growth_recent": "last 4 weeks (or 3 months) vs the 4 before (M1 vitals)",
    "runway_months": "months of cash at current sales and costs; 999 when profitable (M1)",
    "profitable": "gross profit covers fixed costs (M1)",
    "cash": "cash on hand (founder)",
    "monthly_fixed_costs": "fixed costs per month (founder)",
    "gross_margin": "share of a sale left after cost of goods (founder)",
    "min_margin": "founder's floor for pricing moves",
    "stock_cover_weeks": "weeks the stock on hand lasts at recent sales (M1)",
    "supplier_lead_time_weeks": "weeks a new order takes to arrive (founder)",
    "forecast_growth": "next 3 months vs the last 3, from the forecast (M3)",
    "band_width": "forecast range width relative to the middle (M3)",
    "weeks_to_peak": "weeks until the forecast's busiest week, if it is 25%+ above recent sales (M3)",
    "peak_ratio": "busiest forecast week vs recent median (M3)",
    "sales_volatility": "coefficient of variation of the last 12 periods (M1)",
    "market_pulse": "rising / flat / falling for the next 12 weeks (M5 rule)",
    "top_product": "best-selling product (M1)",
    "top_product_share": "its share of recent sales (M1)",
    "market_share": "monthly sales / category size (M7, founder gives category size)",
    "share_change_expected": "predicted change in market share next month (M7 share model, marketplace mode)",
    "rival_count": "number of named rivals (founder)",
    "rival_p_exit": "highest exit probability among rivals (M7)",
    "rival_exit_name": "that rival",
    "rival_p_expansion": "highest expansion probability among rivals (M7)",
    "rival_expanding_name": "that rival",
    "rival_p_price_cut": "rival price-cut probability: base rate only, not predictable (decision 0012)",
    "rival_price_cut_seen": "founder saw a rival cut prices",
}


def _f(name, value, cf=1.0, source=Source.MEASURED, evidence="") -> Fact:
    assert name in FACTS, name
    return Fact(name=name, value=value, certainty=round(float(cf), 3), source=source, evidence=evidence or FACTS[name])


def market_pulse(values: list[float], season: int, h: int) -> tuple[str, float, str] | None:
    """Step 5 verdict: 'same weeks last year' when a year plus a window exists, else the recent slope."""
    y = np.asarray(values, float)
    w = min(12, season)
    if len(y) >= season + w:
        change = (y[len(y) - season:len(y) - season + h].mean() + 1) / (y[len(y) - season - w:len(y) - season].mean() + 1) - 1
        how, cf = "same weeks last year", MARKET_PULSE_CF
    elif len(y) >= 8:
        recent = y[-8:]
        slope = np.polyfit(np.arange(8), recent, 1)[0] if recent.std() > 0 else 0.0
        change, how, cf = slope * h / (recent.mean() + 1), "recent slope", RECENT_SLOPE_CF
    else:
        return None
    label = "rising" if change > TREND_THRESHOLD else "falling" if change < -TREND_THRESHOLD else "flat"
    return label, cf, f"{how}: {change:+.0%}"


def build(got: IntakeResult, vitals: list[Fact], fc: Forecast | None, ctx: BusinessContext) -> list[Fact]:
    s = got.series[0]
    y = np.asarray(s.values, float)
    weekly = s.frequency == "W"
    per_week = (lambda v: v) if weekly else (lambda v: v / WEEKS_PER_MONTH)
    out = [_f("history_weeks", float(len(y) if weekly else len(y) * WEEKS_PER_MONTH)), _f("data_tier", got.tier.value)]

    for v in vitals:
        if v.name == "growth_recent":
            out.append(_f("growth_recent", float(v.value), evidence=v.evidence))
        if v.name == "runway_months":
            profitable = v.value == "profitable"
            out.append(_f("runway_months", 999.0 if profitable else float(v.value), evidence=v.evidence))
            out.append(_f("profitable", profitable))
    for name in ("cash_on_hand", "monthly_fixed_costs", "gross_margin"):
        val = getattr(ctx, name)
        if val is not None:
            out.append(_f("cash" if name == "cash_on_hand" else name, float(val)))
    out.append(_f("min_margin", ctx.min_margin))

    recent = y[-12:]
    if len(recent) >= 4 and recent.mean() > 0:
        out.append(_f("sales_volatility", float(recent.std() / recent.mean())))
    weekly_sales = per_week(float(y[-4:].mean())) if len(y) else 0.0
    if ctx.stock_value is not None and weekly_sales > 0:
        margin = ctx.gross_margin if ctx.gross_margin is not None else 0.30
        cover = ctx.stock_value / (weekly_sales * (1 - margin))
        out.append(_f("stock_cover_weeks", round(cover, 2), 1.0 if ctx.gross_margin is not None else 0.7,
                      evidence=f"stock {ctx.stock_value:,.0f} at cost / weekly sales at cost {weekly_sales * (1 - margin):,.0f}"))
    if ctx.supplier_lead_time_days is not None:
        out.append(_f("supplier_lead_time_weeks", round(ctx.supplier_lead_time_days / 7, 1)))

    if fc is not None:
        h = len(fc.p50)
        past = y[-h:].sum()
        mid, lo, hi = sum(fc.p50), sum(fc.p10), sum(fc.p90)
        band = (hi - lo) / mid if mid > 0 else np.inf
        cf = float(np.clip(1 - band / 2, 0.3, 0.95))
        if past > 0:
            out.append(_f("forecast_growth", round(mid / past - 1, 4), cf, Source.PREDICTED, f"{fc.model}: next {h} vs last {h} periods"))
        out.append(_f("band_width", round(float(band), 3), 1.0, Source.PREDICTED))
        if weekly and len(y) >= 4:
            base = float(np.median(y[-12:])) or 1.0
            p = np.asarray(fc.p50)
            ratio = p.max() / base
            if ratio >= PEAK_RATIO:
                out.append(_f("weeks_to_peak", float(p.argmax() + 1), cf, Source.PREDICTED))
                out.append(_f("peak_ratio", round(float(ratio), 2), cf, Source.PREDICTED))

    pulse = market_pulse(s.values, 52 if weekly else 12, 12 if weekly else 3)
    if pulse:
        out.append(_f("market_pulse", pulse[0], pulse[1], Source.MEASURED, pulse[2]))

    products = [p for p in got.series[1:] if p.product]
    if products:
        n = 12 if weekly else 3
        totals = {p.product: sum(p.values[-n:]) for p in products}
        top = max(totals, key=totals.get)
        all_recent = float(y[-n:].sum())
        if all_recent > 0:
            out.append(_f("top_product", top))
            out.append(_f("top_product_share", round(totals[top] / all_recent, 3)))

    if ctx.market_size_monthly:
        out.append(_f("market_share", round(per_week(float(y[-4:].mean())) * WEEKS_PER_MONTH / ctx.market_size_monthly, 4), 0.8, Source.INFERRED))
    if ctx.competitors:
        out.append(_f("rival_count", float(len(ctx.competitors))))
        out.append(_f("rival_p_exit", RIVAL_BASE_RATES["exit"], RIVAL_BASE_CF, Source.INFERRED, "base rate from Olist sellers; no data on this rival"))
        out.append(_f("rival_exit_name", ctx.competitors[0], RIVAL_BASE_CF, Source.INFERRED))
        out.append(_f("rival_p_expansion", RIVAL_BASE_RATES["expansion"], RIVAL_BASE_CF, Source.INFERRED, "base rate from Olist sellers"))
        out.append(_f("rival_expanding_name", ctx.competitors[0], RIVAL_BASE_CF, Source.INFERRED))
        out.append(_f("rival_p_price_cut", RIVAL_BASE_RATES["price_cut"], RIVAL_BASE_CF, Source.INFERRED, "base rate; not predictable"))
    if ctx.rival_price_cut_seen is not None:
        out.append(_f("rival_price_cut_seen", bool(ctx.rival_price_cut_seen)))
    return out


def marketplace(seller_id: str, category: str, month: str, rivals: int = 3) -> list[Fact]:
    """Marketplace mode: the seller and its rivals are visible (Olist format), so the Step 5
    competitor and share models score them. Returns rival and share facts for that month."""
    from startupiq.models import serve
    rows = serve.olist_rows()
    m = rows[(rows.category.astype(str) == category) & (rows.month.astype(str) == month)]
    me = m[m.seller_id == seller_id]
    if me.empty:
        return []
    others = m[m.seller_id != seller_id].sort_values("share", ascending=False).head(rivals)
    out = [_f("rival_count", float(len(others)), 1.0, Source.MEASURED, f"top {len(others)} rivals by share in {category}, {month}")]
    if len(others):
        p = serve.move_probabilities(others)
        ex, gr = p["p_exit"].idxmax(), p["p_expansion"].idxmax()
        out += [_f("rival_p_exit", round(float(p.loc[ex, "p_exit"]), 3), RIVAL_MODEL_CF, Source.PREDICTED, "competitor random forest (decision 0012)"),
                _f("rival_exit_name", f"Rival {str(others.loc[ex, 'seller_id'])[:6]}", RIVAL_MODEL_CF, Source.PREDICTED),
                _f("rival_p_expansion", round(float(p.loc[gr, "p_expansion"]), 3), RIVAL_MODEL_CF, Source.PREDICTED, "competitor random forest"),
                _f("rival_expanding_name", f"Rival {str(others.loc[gr, 'seller_id'])[:6]}", RIVAL_MODEL_CF, Source.PREDICTED),
                _f("rival_p_price_cut", RIVAL_BASE_RATES["price_cut"], RIVAL_BASE_CF, Source.INFERRED, "base rate; not predictable")]
    share_now = float(me.share.iloc[0])
    share_next = float(serve.next_share(me)[0])
    out += [_f("market_share", round(share_now, 4), 1.0, Source.MEASURED, "seller revenue / category revenue this month"),
            _f("share_change_expected", round(share_next - share_now, 4), SHARE_ALERT_PRECISION, Source.PREDICTED,
               "share LightGBM: MAE 1.55 pct points on test; 69% of predicted drops over 2 points were real drops")]
    return out


def as_frame(facts: list[Fact]) -> pd.DataFrame:
    return pd.DataFrame([{"fact": f.name, "value": f.value, "certainty": f.certainty, "source": f.source.value, "evidence": f.evidence} for f in facts])
