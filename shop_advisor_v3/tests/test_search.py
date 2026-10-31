"""Task 3: state space, simulator and the five search algorithms on small synthetic businesses."""

import numpy as np
import pytest

from startupiq.search.algorithms import ALGORITHMS, astar, bfs, ucs
from startupiq.search.problem import Business, Problem


def shop(**kw) -> Business:
    base = dict(weekly_p50=np.full(12, 10000.0), weekly_p90=np.full(12, 13000.0), cash=150000, monthly_fixed_costs=30000,
                gross_margin=0.40, stock_value=60000, lead_time_weeks=2)
    base.update(kw)
    return Business(**base)


def test_routine_reorders_prevent_stockouts_at_normal_demand():
    p = Problem(shop(stock_value=0))
    assert p.baseline.lost_sales == pytest.approx(0)


def test_demand_above_normal_draws_down_stock_and_restock_helps():
    peak = np.full(12, 10000.0); peak[4:8] = 20000
    p = Problem(shop(weekly_p50=peak, weekly_p90=peak * 1.2, stock_value=5000))
    assert p.baseline.lost_sales > 0
    assert p.simulate(((0, "restock"),)).lost_sales < p.baseline.lost_sales


def test_cost_reduction_raises_profit_and_illegal_move_is_never_offered():
    p = Problem(shop())
    assert p.simulate(((0, "cost_reduction"),)).profit > p.baseline.profit
    offered = {m for plan, _ in p.successors(()) for _, m in plan}
    assert "price_coordination" not in offered and "cost_reduction" in offered


def test_constraints_block_unaffordable_moves():
    p = Problem(shop(cash=1000))
    offered = {m for plan, _ in p.successors(()) for _, m in plan}
    assert "new_product" not in offered and "channel_expansion" not in offered


def test_rule_forbidden_moves_are_excluded():
    p = Problem(shop(), forbidden={"price_cut", "promotion"})
    offered = {m for plan, _ in p.successors(()) for _, m in plan}
    assert not offered & {"price_cut", "promotion"}


def test_canonical_order_generates_each_set_once():
    p = Problem(shop())
    children = [plan for plan, _ in p.successors(((2, "loyalty_offer"),))]
    assert all(c[-1] >= (2, "loyalty_offer") for c in children) and len(children) == len(set(children))


@pytest.mark.parametrize("k", [0.5, 1.5, 2.5])
def test_astar_matches_uniform_cost_and_bfs_uses_fewest_moves(k):
    p0 = Problem(shop())
    gain = max(p0.simulate(((w, m),)).profit - p0.baseline.profit for m in p0.moves for w in (0, 2, 4, 6, 8, 10)
               if not p0.violations(m, 150000, 60000))
    p = Problem(shop(), p0.baseline.profit + k * gain)
    a, u, b = astar(p), ucs(p), bfs(p)
    if u.plan is None:
        assert a.plan is None and b.plan is None
        return
    assert a.cost == pytest.approx(u.cost)
    assert a.expanded <= u.expanded
    assert len(b.plan) <= len(u.plan)
    assert p.is_goal(a.plan) and p.is_goal(b.plan)


def test_heuristic_is_admissible_at_the_root():
    p0 = Problem(shop())
    p = Problem(shop(), p0.baseline.profit + 1.5 * max(p0.simulate(((0, m),)).profit - p0.baseline.profit for m in p0.moves))
    u = ucs(p)
    assert u.plan is not None and p.heuristic(()) <= u.cost + 1e-6


def test_unreachable_target_returns_no_plan_from_every_algorithm():
    p = Problem(shop(), target=1e12)
    for fn in ALGORITHMS.values():
        r = fn(p)
        assert r.plan is None and not r.hit_limit
