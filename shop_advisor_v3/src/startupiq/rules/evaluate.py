"""Evaluate the rule base (Task 2): python -m startupiq.rules.evaluate

1. Agreement with 15 hand-written founder scenarios (tests/fixtures/rules/scenarios.yaml).
2. Rule coverage: which rules ever fire on the scenarios and on real data.
3. Behaviour on real uploads: 60 Rossmann stores and 20 Olist sellers (marketplace mode,
   where the Step 5 competitor and share models supply rival and share facts).
Writes docs/results/rules.md and rules.json.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from startupiq.contracts import BusinessContext, Fact
from startupiq.evaluate import report
from startupiq.pipeline import analyze
from startupiq.rules import facts as fx
from startupiq.rules import loader

SCENARIOS = Path("tests/fixtures/rules/scenarios.yaml")
ROSSMANN_STORES = 60
OLIST_SELLERS = 20
OLIST_MONTH = "2018-05"  # last feature month of the test split


def scenario_agreement(engine) -> tuple[pd.DataFrame, Counter]:
    rows, fired = [], Counter()
    for name, spec in yaml.safe_load(SCENARIOS.read_text(encoding="utf-8")).items():
        res = engine.run([Fact(name=k, value=v) for k, v in spec["facts"].items()])
        fired.update(f.rule.id for f in res.trace)
        checks = []
        for m in spec.get("recommend", []):
            checks.append(m in res.recommendations)
        for m in spec.get("not_recommend", []):
            checks.append(m not in res.recommendations)
        for kind, have in (("warn", res.warnings), ("forbid", res.forbidden), ("diagnose", res.diagnoses), ("risk", res.risks)):
            checks += [k in have for k in spec.get(kind, [])]
        rows.append({"scenario": name, "expectations": len(checks), "met": sum(checks),
                     "top recommendation": next(iter(res.recommendations), "-")})
    return pd.DataFrame(rows), fired


def rossmann_runs(fired: Counter) -> pd.DataFrame:
    d = pd.read_parquet("data/processed/rossmann_daily.parquet", columns=["Store", "Date", "Sales", "Open"])
    stores = sorted(d.Store.unique())[:: max(1, len(d.Store.unique()) // ROSSMANN_STORES)][:ROSSMANN_STORES]
    rows = []
    for s in stores:
        up = d[(d.Store == s) & (d.Open == 1)][["Date", "Sales"]]
        weekly = up.Sales.sum() / up.Date.dt.to_period("W").nunique()
        # Cash and costs are not in the data: assume a store with 3 months of fixed costs in cash and a 30% margin,
        # fixed costs at 25% of sales, stock worth 3 weeks at cost and a 2-week supplier: a plausible, stated assumption.
        ctx = BusinessContext(cash_on_hand=weekly * 52 / 12 * 0.25 * 3, monthly_fixed_costs=weekly * 52 / 12 * 0.25,
                              gross_margin=0.30, stock_value=weekly * 3 * 0.7, supplier_lead_time_days=14, competitors=["nearest rival"])
        rep = analyze(up, ctx)
        rows.append(_summary(rep, f"rossmann_{s}"))
        fired.update(t.split(" ")[1] for t in rep.trace if t.startswith("["))
    return pd.DataFrame(rows)


def olist_runs(fired: Counter) -> pd.DataFrame:
    items = pd.read_parquet("data/processed/olist_items.parquet")
    from startupiq.models import serve
    rows_all = serve.olist_rows()
    month_rows = rows_all[rows_all.month.astype(str) == OLIST_MONTH]
    top = month_rows.sort_values("log_revenue", ascending=False).drop_duplicates("seller_id").head(OLIST_SELLERS)
    rows = []
    for _, r in top.iterrows():
        cat, seller = str(r.category), r.seller_id
        up = items[(items.seller_id == seller) & (items.category == cat) & (items.month <= OLIST_MONTH)][["order_purchase_timestamp", "price"]]
        up = up.rename(columns={"order_purchase_timestamp": "date", "price": "sales"})
        extra = fx.marketplace(seller, cat, OLIST_MONTH)
        rep = analyze(up, BusinessContext(gross_margin=0.30), extra_facts=extra)
        rows.append(_summary(rep, f"olist_{seller[:6]}_{cat}"))
        fired.update(t.split(" ")[1] for t in rep.trace if t.startswith("["))
    return pd.DataFrame(rows)


def _summary(rep, name) -> dict:
    cards = {c.key: c for c in rep.cards}
    nm = cards.get("next_move")
    w = cards.get("warnings")
    return {"upload": name, "tier": rep.tier.value,
            "warnings": ", ".join((w.details.get("warnings") or {}).keys()) if w else "-",
            "diagnoses": ", ".join((w.details.get("diagnoses") or {}).keys()) if w else "-",
            "recommendations": ", ".join(r.action for r in rep.recommendations) or "-",
            "blocked by safety gate": ", ".join(nm.details.get("blocked", {}).keys()) if nm else "-"}


def run() -> dict:
    t0 = time.time()
    rules = loader.load()
    engine = loader.engine()
    scen, fired_s = scenario_agreement(engine)
    fired_real: Counter = Counter()
    ross = rossmann_runs(fired_real)
    oli = olist_runs(fired_real)
    cov = pd.DataFrame([{"rule": r.id, "name": r.name, "group": r.group, "fired in scenarios": fired_s[r.id],
                         "fired on real uploads": fired_real[r.id]} for r in rules])
    agree = scen.met.sum() / scen.expectations.sum()
    rec_counts = Counter(x.strip() for col in (ross.recommendations, oli.recommendations) for v in col for x in v.split(",") if x.strip() != "-")
    out = {"rules": len(rules), "groups": dict(Counter(r.group for r in rules)), "scenario_agreement": round(float(agree), 4),
           "scenario_expectations": int(scen.expectations.sum()), "coverage_scenarios": round(float((cov["fired in scenarios"] > 0).mean()), 3),
           "coverage_real": round(float((cov["fired on real uploads"] > 0).mean()), 3), "real_recommendations": dict(rec_counts),
           "seconds": round(time.time() - t0)}
    report.write("rules", "Rule base evaluation (Task 2)", [
        ("Summary", f"{len(rules)} rules in {len(out['groups'])} groups {out['groups']}. Scenario agreement **{agree:.1%}** "
                    f"({scen.met.sum()} of {scen.expectations.sum()} expectations). Rules that fired at least once: "
                    f"{out['coverage_scenarios']:.0%} on the scenarios, {out['coverage_real']:.0%} on {len(ross) + len(oli)} real uploads."),
        ("Founder scenarios", report.table(scen)),
        ("How often each recommendation appeared on real uploads", report.table(pd.DataFrame(rec_counts.most_common(), columns=["recommendation", "uploads"]))),
        ("Rossmann stores (assumed cash, costs and stock; see code)", report.table(ross)),
        ("Olist sellers in marketplace mode (rival and share facts from the Step 5 models)", report.table(oli)),
        ("Rule coverage", report.table(cov)),
    ], out, command="python -m startupiq.rules.evaluate")
    return out


if __name__ == "__main__":
    print(run())
