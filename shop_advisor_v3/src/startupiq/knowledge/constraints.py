"""Domain constraints and the consistency checker (Task 1).

Each constraint is a named predicate over the knowledge base, optionally about one
proposed move. `check()` returns the violated constraints with a plain-language reason.
Constraint names are the ones moves list under `requires` in configs/moves.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from startupiq.knowledge.frames import Frame

PROMO_COVER_WEEKS = 2   # a promotion needs at least two weeks of stock on hand
EXPANSION_BUFFER_WEEKS = 2  # a new channel needs lead time + 2 weeks of cover


@dataclass
class Constraint:
    name: str
    statement: str
    test: Callable[[Frame, Frame | None], str | None]  # returns a reason when violated, else None


def move_cash(biz: Frame, move: Frame) -> float:
    """Cash a move needs now, from the moves catalogue and the business's sales."""
    monthly = biz.get("monthly_sales") or 0.0
    if move.get("cost_rule") == "weeks_of_sales_at_cost":
        return move.get("weeks_ordered") * (biz.get("weekly_sales") or 0) * (1 - biz.get("gross_margin"))
    return (move.get("cost") or 0.0) * monthly


def _cash_covers(biz, move):
    if move is None or biz.get("cash") is None:
        return None
    need = move_cash(biz, move)
    return None if need <= biz.get("cash") else f"needs {need:,.0f} but cash is {biz.get('cash'):,.0f}"


def _margin_floor(biz, move):
    if move is None:
        return None
    after = biz.get("gross_margin") + (move.get("margin_change") or 0.0)
    return None if after >= biz.get("min_margin") else f"margin would fall to {after:.0%}, below the {biz.get('min_margin'):.0%} floor"


def _lead_time_known(biz, move):
    _, how = biz.get("supplier_lead_time_weeks", explain=True)
    return None if how.startswith("value") else "supplier lead time not given (the 2-week default would be a guess)"


def _runway6(biz, move):
    r = biz.get("runway_months")
    if r is None:
        return "runway unknown (cash or costs not given)"
    return None if r >= 6 else f"runway is {r:.1f} months; this move needs at least 6"


def _stock_promo(biz, move):
    cover = biz.get("stock_cover_weeks")
    if cover is None:
        return None  # unknown stock is a warning for the rule engine, not a hard violation
    return None if cover >= PROMO_COVER_WEEKS * (move.get("demand_lift") or 1) else f"only {cover:.1f} weeks of stock; a promotion would sell out"


def _stock_expansion(biz, move):
    cover = biz.get("stock_cover_weeks")
    need = biz.get("supplier_lead_time_weeks") + EXPANSION_BUFFER_WEEKS
    if cover is None:
        return "stock on hand unknown"
    return None if cover >= need else f"{cover:.1f} weeks of stock; a new channel needs {need} (lead time + {EXPANSION_BUFFER_WEEKS})"


def _legal(biz, move):
    return None if move is None or move.get("legal") else "illegal: agreeing prices with a competitor is a cartel under competition law"


CONSTRAINTS = {c.name: c for c in [
    Constraint("cash_covers_cost", "A move's cash cost must not exceed cash on hand.", _cash_covers),
    Constraint("margin_stays_above_floor", "A pricing move must keep gross margin at or above the business's floor.", _margin_floor),
    Constraint("lead_time_known", "Restocking needs a known supplier lead time.", _lead_time_known),
    Constraint("runway_at_least_6_months", "Moves that take a quarter to pay back need 6+ months of runway.", _runway6),
    Constraint("stock_covers_promotion", "A promotion needs enough stock for the extra demand.", _stock_promo),
    Constraint("stock_covers_expansion", "A new channel needs stock for the lead time plus a buffer.", _stock_expansion),
    Constraint("legal", "No move may break competition law (no price coordination).", _legal),
]}
