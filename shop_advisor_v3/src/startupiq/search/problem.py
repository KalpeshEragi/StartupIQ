"""The planning problem as a state space (Task 3, decision 0008).

State     a plan: a sorted tuple of (start_week, move) pairs, at most MAX_MOVES long.
Initial   the empty plan (keep doing what you do).
Actions   add one move at a decision week (0, 2, ..., 10) not earlier than the last one in the
          plan (so each set of moves is generated once), if the knowledge model's constraints
          allow it at that point (cash, margin floor, stock cover, legality, runway ...).
Transition  the simulator plays the plan forward for 12 weeks on the forecast: demand lift,
          margin change, stock (the usual weekly reorder keeps arriving; a restock move orders extra,
          which arrives after the supplier lead time), fixed costs, cash.
Goal      12-week gross profit >= target and cash never below zero.
Cost      what the plan gives up: cash paid for moves + margin given away by price cuts.
Heuristic fractional-knapsack bound: cover the remaining profit gap with the remaining moves,
          best profit per unit cost first, fractions allowed. Admissible if moves do not reinforce
          each other beyond their separate effects; we check A* against uniform-cost search instead
          of assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from startupiq.knowledge.constraints import CONSTRAINTS, move_cash
from startupiq.knowledge.ontology import build_classes, move_names

HORIZON = 12
DECISION_WEEKS = (0, 2, 4, 6, 8, 10)
MAX_MOVES = 4
REPEATABLE = {"promotion": 2, "restock": 2}
WEEKS_PER_MONTH = 52 / 12
TARGET_UPLIFT = 0.03  # default goal: baseline profit + 3% of 12-week sales
HOLDING_COST = 0.05   # cost of holding ordered stock for a quarter: storage and tied-up cash (assumption)


@dataclass
class Business:
    weekly_p50: np.ndarray            # 12 weeks of forecast sales (value at today's prices)
    weekly_p90: np.ndarray
    cash: float
    monthly_fixed_costs: float
    gross_margin: float
    min_margin: float = 0.10
    stock_value: float | None = None  # at cost; None = not tracked (no stock cap)
    lead_time_weeks: float | None = None


@dataclass
class Outcome:
    profit: float
    min_cash: float
    lost_sales: float
    end_cash: float
    weekly_profit: list[float] = field(default_factory=list)


class Problem:
    def __init__(self, biz: Business, target: float | None = None, forbidden: set[str] | frozenset = frozenset()):
        """forbidden: moves the rule engine ruled out (decision 0013); the planner never considers them."""
        self.biz = biz
        self.classes = build_classes()
        self.moves = [m for m in move_names(self.classes) if m != "hold" and m not in forbidden]
        self.base_weekly = float(np.mean(biz.weekly_p50[:4]))
        self.baseline = self.simulate(())
        sales12 = float(np.sum(biz.weekly_p50))
        self.target = target if target is not None else self.baseline.profit + TARGET_UPLIFT * sales12
        self._ratios = self._move_ratios()

    # ---------- the knowledge model, per state ----------
    def _frame(self, cash: float, stock: float | None):
        b = self.biz
        f = self.classes["SmallSeller"].instance("you", cash=cash, monthly_fixed_costs=b.monthly_fixed_costs, gross_margin=b.gross_margin,
                                                 min_margin=b.min_margin, weekly_sales=self.base_weekly)
        if stock is not None:
            f.set("stock_value", stock)
        if b.lead_time_weeks is not None:
            f.set("supplier_lead_time_weeks", b.lead_time_weeks)
        return f

    def violations(self, move: str, cash: float, stock: float | None) -> list[str]:
        mv = self.classes[move]
        biz = self._frame(cash, stock)
        return [r for r in mv.get("requires") or [] if CONSTRAINTS[r].test(biz, mv)]

    # ---------- simulator ----------
    def _lead(self, move: str) -> float:
        lt = self.classes[move].get("lead_time_weeks")
        return (self.biz.lead_time_weeks or 2) if lt == "supplier" else float(lt or 0)

    def cost(self, move: str) -> float:
        """What a move gives up: cash spent (for stock, only the cost of holding it, because the cash
        turns back into sales) plus margin given away on baseline sales while a price move lasts."""
        mv = self.classes[move]
        cash = move_cash(self._frame(self.biz.cash, self.biz.stock_value), mv)
        if mv.get("cost_rule") == "weeks_of_sales_at_cost":
            cash *= HOLDING_COST
        give = abs(mv.get("margin_change") or 0.0) * self.base_weekly * (mv.get("duration_weeks") or 0)
        return float(cash + give)

    def simulate(self, plan: tuple, demand: np.ndarray | None = None) -> Outcome:
        b = self.biz
        demand = b.weekly_p50 if demand is None else demand
        cash, stock = b.cash, b.stock_value
        weekly_fixed = b.monthly_fixed_costs / WEEKS_PER_MONTH
        routine = self.base_weekly * (1 - b.gross_margin)  # the shop's usual weekly reorder keeps arriving
        profit, min_cash, lost, wp = 0.0, cash, 0.0, []
        frame = self._frame(b.cash, b.stock_value)
        for w in range(HORIZON):
            lift, margin, fixed = 1.0, b.gross_margin, weekly_fixed
            for start, m in plan:
                mv = self.classes[m]
                if w == start:
                    cash -= move_cash(frame, mv)
                on = start + self._lead(m)
                dur = mv.get("duration_weeks") or 0
                active = on <= w < (on + dur if dur else HORIZON)
                if m == "restock" and w == int(np.ceil(on)) and stock is not None:
                    stock += mv.get("weeks_ordered") * self.base_weekly * (1 - b.gross_margin)
                if active and m != "restock":
                    lift *= mv.get("demand_lift") or 1.0
                    margin += mv.get("margin_change") or 0.0
                    if mv.get("fixed_cost_change"):
                        fixed *= 1 + mv.get("fixed_cost_change")
            want = demand[w] * lift
            if stock is not None:
                stock += routine  # routine replenishment (its cost is inside the gross margin); a restock move adds to it
                sold = min(want, max(stock, 0.0) / max(1 - margin, 1e-6))
                stock -= sold * (1 - margin)
            else:
                sold = want
            lost += want - sold
            gp = sold * margin - fixed
            profit += gp
            cash += gp
            min_cash = min(min_cash, cash)
            wp.append(gp)
        return Outcome(profit, min_cash, lost, cash, wp)

    # ---------- search interface ----------
    def is_goal(self, plan: tuple) -> bool:
        o = self._outcome(plan)
        return o.profit >= self.target and o.min_cash >= 0

    @lru_cache(maxsize=200_000)
    def _outcome(self, plan: tuple) -> Outcome:
        return self.simulate(plan)

    def _state_at(self, plan: tuple, week: int) -> tuple[float, float | None]:
        """Cash and stock just before `week` if the plan so far is followed (used to check constraints there)."""
        o = self._outcome(plan)
        cash = self.biz.cash + sum(o.weekly_profit[:week]) - sum(move_cash(self._frame(self.biz.cash, self.biz.stock_value), self.classes[m])
                                                                 for s, m in plan if s < week)
        stock = None
        if self.biz.stock_value is not None:  # usual reorders cover usual sales, so only demand above the usual level draws stock down
            above = sum(max(0.0, d - self.base_weekly) for d in self.biz.weekly_p50[:week])
            stock = max(0.0, self.biz.stock_value - above * (1 - self.biz.gross_margin))
        return cash, stock

    def successors(self, plan: tuple):
        if len(plan) >= MAX_MOVES:
            return
        last = plan[-1] if plan else (0, "")
        for week in DECISION_WEEKS:
            if week < last[0]:
                continue
            cash, stock = self._state_at(plan, week)
            for m in self.moves:
                if (week, m) <= last:
                    continue  # canonical order: each set of moves is generated once
                if sum(1 for _, x in plan if x == m) >= REPEATABLE.get(m, 1):
                    continue
                if self.violations(m, cash, stock):
                    continue
                yield tuple(sorted(plan + ((week, m),))), self.cost(m)

    def heuristic(self, plan: tuple) -> float:
        """Fractional-knapsack lower bound on the cost still needed: cover the remaining profit gap with the
        moves still available, cheapest per unit of profit first, allowing fractions of a move. Gains are
        measured with the same simulator and demand (P50) as the goal test. Never more than the true remaining
        cost if moves do not gain more together than apart; A* is checked against uniform-cost search for that."""
        gap = self.target - self._outcome(plan).profit
        if gap <= 0:
            return 0.0
        used = {m: sum(1 for _, x in plan if x == m) for m in self.moves}
        room = MAX_MOVES - len(plan)
        need = 0.0
        for m, gain, cost in self._ratios:
            uses = min(REPEATABLE.get(m, 1) - used[m], room)
            for _ in range(max(0, uses)):
                if gain >= gap:
                    return need + cost * gap / gain
                need, gap, room = need + cost, gap - gain, room - 1
                if room <= 0:
                    return need  # still short: any completion costs at least this much
        return need

    def _move_ratios(self) -> list[tuple[str, float, float]]:
        """(move, best stand-alone profit gain, cost), best gain per unit of cost first."""
        base = self.baseline.profit
        out = []
        for m in self.moves:
            c = self.cost(m)
            gain = max(self.simulate(((w, m),)).profit - base for w in DECISION_WEEKS)
            if c > 0 and gain > 0:
                out.append((m, gain, c))
        return sorted(out, key=lambda x: -x[1] / x[2])

    def describe(self, plan: tuple) -> list[str]:
        return [f"week {w}: {self.classes[m].get('label')}" for w, m in plan] or ["keep things as they are"]
