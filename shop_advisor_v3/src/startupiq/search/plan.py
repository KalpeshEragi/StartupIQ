"""Planner for the product (M9): from a founder's forecast and finances to the cheapest plan that
reaches the default profit target, found by A* (optimal in every benchmark case, decision 0014)."""

from __future__ import annotations

import numpy as np

from startupiq.contracts import BusinessContext, Forecast, SalesSeries
from startupiq.search.algorithms import astar
from startupiq.search.problem import TARGET_UPLIFT, Business, Problem


def business(series: SalesSeries, fc: Forecast, ctx: BusinessContext) -> Business | None:
    """None when the plan cannot be built honestly: needs weekly data and the founder's cash, costs and margin."""
    if series.frequency != "W" or None in (ctx.cash_on_hand, ctx.monthly_fixed_costs, ctx.gross_margin) or len(fc.p50) < 12:
        return None
    return Business(np.asarray(fc.p50[:12], float), np.asarray(fc.p90[:12], float), cash=ctx.cash_on_hand,
                    monthly_fixed_costs=ctx.monthly_fixed_costs, gross_margin=ctx.gross_margin, min_margin=ctx.min_margin,
                    stock_value=ctx.stock_value, lead_time_weeks=None if ctx.supplier_lead_time_days is None else ctx.supplier_lead_time_days / 7)


def plan(series: SalesSeries, fc: Forecast, ctx: BusinessContext, forbidden: set[str]) -> dict | None:
    biz = business(series, fc, ctx)
    if biz is None:
        return None
    p = Problem(biz, forbidden=forbidden)
    r = astar(p)
    out = {"target_profit_12w": round(p.target), "baseline_profit_12w": round(p.baseline.profit),
           "goal": f"12-week gross profit at least {p.target:,.0f} (doing nothing: {p.baseline.profit:,.0f}; target adds {TARGET_UPLIFT:.0%} of sales) with cash never below zero",
           "nodes_expanded": r.expanded, "excluded_by_rules": sorted(forbidden)}
    if r.plan is None:
        out.update(found=False, steps=[], note="No combination of up to 4 allowed moves reaches the target within the cash available.")
        return out
    o = p.simulate(r.plan)
    out.update(found=True, steps=p.describe(r.plan), cost=round(r.cost), projected_profit_12w=round(o.profit), lowest_cash=round(o.min_cash),
               note="Effects of moves are assumptions listed in configs/moves.yaml; the plan is a starting point, not a promise.")
    return out
