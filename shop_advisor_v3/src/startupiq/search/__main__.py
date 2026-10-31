"""Task 3 comparison: python -m startupiq.search

Four businesses built from real uploads (finances stated as assumptions where the data has
none) x three profit targets x five algorithms. Targets are set relative to the best single
move for that business (0.5x, 1.5x, 2.5x its profit gain above doing nothing), so the easy one
needs one move and the hard one needs several: that is what separates the algorithms. Uniform-cost search is the exhaustive optimal reference; A* is checked against it.
Writes docs/results/planner.md and .json.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from startupiq.evaluate import report
from startupiq.intake import read_file, read_table
from startupiq.models import forecast_baseline
from startupiq.search.algorithms import ALGORITHMS
from startupiq.search.problem import Business, Problem

TARGETS = {"easy (0.5x best single move)": 0.5, "medium (1.5x)": 1.5, "hard (2.5x)": 2.5}
PACK = Path("tests/fixtures/intake/files")


def business_from(series, cash_months: float, fixed_share: float, margin: float, stock_weeks: float, lead_weeks: float) -> Business:
    fc = forecast_baseline.forecast(series)
    weekly = float(np.mean(series.values[-4:]))
    monthly = weekly * 52 / 12
    return Business(np.asarray(fc.p50), np.asarray(fc.p90), cash=cash_months * monthly * fixed_share, monthly_fixed_costs=monthly * fixed_share,
                    gross_margin=margin, stock_value=stock_weeks * weekly * (1 - margin), lead_time_weeks=lead_weeks)


def scenarios() -> dict[str, tuple[Business, str]]:
    out = {}
    tea = read_file(PACK / "04_title_rows.xlsx").series[0]
    out["Tea stall (synthetic pack file 04)"] = (business_from(tea, 1.5, 0.35, 0.40, 1.5, 1.5),
                                                 "1.5 months of fixed costs in cash, fixed costs 35% of sales, margin 40%, 1.5 weeks of stock, supplier 1.5 weeks")
    d = pd.read_parquet("data/processed/rossmann_daily.parquet", columns=["Store", "Date", "Sales", "Open"])
    store = read_table(d[(d.Store == 1) & (d.Open == 1)][["Date", "Sales"]]).series[0]
    out["Rossmann store 1"] = (business_from(store, 3, 0.25, 0.30, 3, 2), "3 months of fixed costs in cash, fixed costs 25% of sales, margin 30%, 3 weeks of stock, supplier 2 weeks")
    uci = pd.read_parquet("data/processed/uci_total_weekly.parquet")
    up = read_table(uci.rename(columns={"week": "date", "revenue": "sales"})[["date", "sales"]]).series[0]
    out["UCI wholesaler (all products)"] = (business_from(up, 2, 0.20, 0.35, 6, 4), "2 months of fixed costs in cash, fixed costs 20% of sales, margin 35%, 6 weeks of stock, supplier 4 weeks")
    items = pd.read_parquet("data/processed/olist_items.parquet")
    top = items.groupby("seller_id").price.sum().idxmax()
    ol = read_table(items[items.seller_id == top][["order_purchase_timestamp", "price"]].rename(columns={"order_purchase_timestamp": "date", "price": "sales"})).series[0]
    out["Olist top seller"] = (business_from(ol, 2, 0.20, 0.25, 4, 2), "2 months of fixed costs in cash, fixed costs 20% of sales, margin 25%, 4 weeks of stock, supplier 2 weeks")
    return out


def run() -> dict:
    t0 = time.time()
    rows, cases = [], []
    for name, (biz, assumptions) in scenarios().items():
        p0 = Problem(biz)
        best_gain = max(p0.simulate(((w, m),)).profit - p0.baseline.profit for m in p0.moves for w in (0, 2, 4, 6, 8, 10)
                        if not p0.violations(m, biz.cash, biz.stock_value))
        for label, k in TARGETS.items():
            target = p0.baseline.profit + k * best_gain
            p = Problem(biz, target)
            res = {k: fn(p) for k, fn in ALGORITHMS.items()}
            ref = res["ucs"].cost
            for r in res.values():
                rows.append({"business": name, "target": label, "algorithm": r.algorithm,
                             "found": "yes" if r.plan is not None else ("node limit" if r.hit_limit else "no plan exists"),
                             "cost": None if r.cost is None else round(r.cost), "moves": None if r.plan is None else len(r.plan),
                             "optimal": "-" if r.cost is None or ref is None else ("yes" if abs(r.cost - ref) < 1e-6 else f"no (+{r.cost - ref:,.0f})"),
                             "expanded": r.expanded, "generated": r.generated, "ms": round(r.ms),
                             "plan": "; ".join(p.describe(r.plan)) if r.plan is not None else "-"})
            cases.append({"business": name, "assumptions": assumptions, "target": label, "target_profit": round(target),
                          "best_single_move_gain": round(best_gain), "baseline_profit": round(p.baseline.profit),
                          "baseline_lost_sales": round(p.baseline.lost_sales), "astar_matches_ucs": res["astar"].cost == ref or (ref is None and res["astar"].cost is None)})
    df = pd.DataFrame(rows)
    agg = df[df.found == "yes"].groupby("algorithm").agg(found=("found", "size"), mean_expanded=("expanded", "mean"), mean_ms=("ms", "mean"),
                                                        optimal=("optimal", lambda s: int((s == "yes").sum()))).round(1).reset_index()
    agg.insert(1, "problems", len(cases))
    mismatches = [c for c in cases if not c["astar_matches_ucs"]]
    out = {"rows": rows, "cases": cases, "summary": agg.to_dict("records"), "astar_mismatches": len(mismatches), "seconds": round(time.time() - t0)}
    report.write("planner", "Planner: search over business moves (Task 3)", [
        ("Problem", "State = a plan of up to 4 moves placed at weeks 0, 2, ..., 10; goal = 12-week gross profit at or above the target with cash never below zero; "
                    "cost = cash spent (5% holding cost for stock) + margin given away. Moves and constraints come from the knowledge model (configs/moves.yaml, decision 0007). "
                    "The forecast is the product's Auto forecaster."),
        ("Summary over all problems", report.table(agg) + f"\n\nA* returned a costlier plan than uniform-cost search in {len(mismatches)} of {len(cases)} problems."),
        ("Businesses and assumptions", report.table(pd.DataFrame(cases)[["business", "assumptions", "baseline_profit", "baseline_lost_sales", "best_single_move_gain"]].drop_duplicates("business"))),
        ("Every run", report.table(df)),
    ], out, command="python -m startupiq.search")
    return out


if __name__ == "__main__":
    r = run()
    print(r["summary"], "A* mismatches:", r["astar_mismatches"], f"{r['seconds']}s")
