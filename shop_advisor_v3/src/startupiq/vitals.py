"""M1 vitals: plain arithmetic on the user's own numbers. No model, by design.

Why no ML here: devdocs/decisions/0001-technique-per-module.md.
"""

from __future__ import annotations

from startupiq.contracts import BusinessContext, Fact, SalesSeries, Source

WEEKS_PER_MONTH = 52 / 12
WINDOW = {"W": 4, "M": 3}  # compare the last 4 weeks, or the last 3 months, with the window before


def growth(series: SalesSeries) -> float | None:
    """Last window vs the window before, as a fraction. None if not enough history."""
    n = WINDOW.get(series.frequency, 4)
    v = series.values
    if len(v) < 2 * n or sum(v[-2 * n:-n]) == 0:
        return None
    return sum(v[-n:]) / sum(v[-2 * n:-n]) - 1


def monthly_sales(series: SalesSeries) -> float:
    n = WINDOW.get(series.frequency, 4)
    recent = sum(series.values[-n:]) / n
    return recent * WEEKS_PER_MONTH if series.frequency == "W" else recent


def compute(series: SalesSeries, ctx: BusinessContext) -> list[Fact]:
    facts: list[Fact] = []
    n = WINDOW.get(series.frequency, 4)
    g = growth(series)
    if g is not None:
        unit = "weeks" if series.frequency == "W" else "months"
        facts.append(Fact(name="growth_recent", value=round(g, 4),
                          evidence=f"sum(last {n} {unit}) / sum(previous {n} {unit}) - 1"))
    if ctx.cash_on_hand is not None and ctx.monthly_fixed_costs:
        margin = ctx.gross_margin if ctx.gross_margin is not None else 0.0
        burn = ctx.monthly_fixed_costs - monthly_sales(series) * margin
        if burn > 0:
            facts.append(Fact(
                name="runway_months",
                value=round(ctx.cash_on_hand / burn, 1),
                evidence=f"cash {ctx.cash_on_hand:,.0f} / monthly burn {burn:,.0f} (fixed costs minus gross profit)",
            ))
        else:
            facts.append(Fact(name="runway_months", value="profitable", evidence="gross profit covers fixed costs"))
    for f in facts:
        f.source = Source.MEASURED
    return facts
