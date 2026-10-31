"""analyze(): upload in, founder's briefing out (Task 9 entry point).

Each stage is a plain function over the types in contracts.py. Stages not built
yet return a locked card with Source.STUB so the UI and tests work end to end
from day one. Stage order and data flow: devdocs/architecture.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from startupiq import advisor, category, intake, segment, vitals
from startupiq.contracts import BusinessContext, Card, Fact, IntakeResult, Report, Source, Tier
from startupiq.models import forecast_baseline

FORECAST_TIERS = {Tier.BASIC, Tier.STANDARD, Tier.FULL}
PERIOD_NAME = {"W": "weeks", "M": "months", "D": "days"}


def _locked(key: str, title: str, reason: str) -> Card:
    return Card(key=key, title=title, headline="Not available yet", source=Source.STUB, locked_reason=reason)


def _intake_card(got: IntakeResult, headline: str) -> Card:
    return Card(key="intake", title="Your data", headline=headline, source=Source.MEASURED, details={
        "tier": got.tier.value, "column_map": got.column_map, "confidence": got.confidence,
        "needs_confirmation": got.needs_confirmation, "messages": got.messages, "cleaning_log": got.cleaning_log,
        "layout": got.layout, "granularity": got.granularity, "sheet": got.sheet})


def analyze(table: pd.DataFrame, ctx: BusinessContext | None = None, column_map: dict[str, str] | None = None,
            extra_facts: list[Fact] | None = None) -> Report:
    return analyze_intake(intake.read_table(table, column_map), ctx, extra_facts)


def analyze_file(path: str | Path, ctx: BusinessContext | None = None, column_map: dict[str, str] | None = None,
                 extra_facts: list[Fact] | None = None) -> Report:
    return analyze_intake(intake.read_file(path, column_map), ctx, extra_facts)


def analyze_intake(got: IntakeResult, ctx: BusinessContext | None = None, extra_facts: list[Fact] | None = None) -> Report:
    """extra_facts: model-scored facts from marketplace mode (rules.facts.marketplace), which replace base rates."""
    ctx = ctx or BusinessContext()
    trace = [f"intake: tier={got.tier.value}, columns={got.column_map}, confirm={got.needs_confirmation}"]
    if got.tier == Tier.REFUSED:
        return Report(tier=got.tier, cards=[_intake_card(got, " ".join(got.messages))], trace=trace)

    series = got.series[0]
    facts = vitals.compute(series, ctx)
    trace.append(f"vitals: {[f.name for f in facts]}")

    cards = [_intake_card(got, f"{len(series.values)} {PERIOD_NAME[series.frequency]} read")]

    sig = pd.DataFrame(got.signals) if got.signals else pd.DataFrame({"date": series.dates, "sales": series.values})
    kind = category.classify(sig.assign(date=pd.to_datetime(sig["date"])))
    if kind is not None:
        trace.append(f"business type: {kind['best_guess']} ({kind['confidence']:.0%}, {kind['model']}), from {', '.join(['sales'] + kind['inputs'])}")
        cards.append(Card(key="category", title="Your business type", source=Source.PREDICTED,
                          headline=(f"{kind['category']} ({kind['confidence']:.0%} sure)" if kind["category"]
                                    else f"Not sure; closest is {kind['best_guess']} ({kind['confidence']:.0%})"), details=kind))

    seg = segment.assign(series.values) if series.frequency == "W" else None
    if seg is not None:
        trace.append(f"segment: {seg['segment']} ({seg['confidence']:.0%}), unusual={seg['unusual']}")
        cards.append(Card(key="segment", title="Your business profile", source=Source.INFERRED,
                          headline=f"{seg['segment']}: {seg['meaning']}", details=seg))

    fc = None
    if got.tier in FORECAST_TIERS:
        fc = forecast_baseline.forecast(series, kind["vector"] if kind else None)
        trace.append(f"forecast: model={fc.model}")
        total = sum(fc.p50)
        cards.append(Card(key="forecast", title="Your next 3 months",
                          headline=f"About {total:,.0f} expected over {len(fc.dates)} {PERIOD_NAME[series.frequency]} "
                                   f"(likely range {sum(fc.p10):,.0f} to {sum(fc.p90):,.0f})",  # not "80%": band coverage is 0.54-0.70 on test (decision 0012)
                          source=fc.source, details={**fc.model_dump(mode="json"), "history": {
                              "dates": [d.isoformat() for d in series.dates[-52:]], "values": series.values[-52:]},
                              "last_total": sum(series.values[-len(fc.p50):]), "expected_total": total, "periods": len(series.values)}))
    else:
        cards.append(_locked("forecast", "Your next 3 months", "Need at least 8 weeks of sales to forecast."))

    money = {f.name: f.value for f in facts}
    cards.append(Card(key="money", title="Your money", source=Source.MEASURED,
                      headline=("Profitable: gross profit covers fixed costs" if money.get("runway_months") == "profitable"
                                else f"Runway: {money['runway_months']} months") if "runway_months" in money
                      else "Add cash and monthly costs to see runway",
                      details={"facts": [f.model_dump() for f in facts]}))

    rule_cards, recs, derived, rule_trace = advisor.advise(got, facts, fc, ctx, extra_facts)
    cards += rule_cards
    return Report(tier=got.tier, cards=cards, facts=derived, recommendations=recs, trace=trace + rule_trace)
