"""M8 diagnosis + M10 safety gate: facts -> rules -> constraint check -> cards.

1. rules.facts builds facts from intake, vitals, forecast and Step 5 models.
2. The rule engine forward-chains to warnings, diagnoses, recommendations, forbids, risks.
3. Every recommended move is checked against the knowledge model's constraints
   (knowledge/constraints.py); a move that breaks one is blocked with the reason.
4. Backward chaining explains why common moves were not recommended.
"""

from __future__ import annotations

from startupiq.contracts import BusinessContext, Card, Fact, Forecast, IntakeResult, Recommendation, Source
from startupiq.knowledge.kb import KnowledgeBase
from startupiq.rules import facts as fx
from startupiq.rules import loader
from startupiq.search import plan as planner

EXPLAIN_IF_ABSENT = ("promotion", "channel_expansion", "price_cut", "restock")
DATA_NOTES = ("no_cash_inputs", "no_stock_input", "no_rivals", "need_finances")  # 'add this to unlock more', not a business problem
_ENGINE = None


def engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = loader.engine()
    return _ENGINE


def knowledge(ctx: BusinessContext, weekly_sales: float, name: str = "you") -> KnowledgeBase:
    """Knowledge base with the founder's business as an instance frame and rivals linked by competes_with."""
    kb = KnowledgeBase()
    kb.add("SmallSeller", name, cash=ctx.cash_on_hand, monthly_fixed_costs=ctx.monthly_fixed_costs, gross_margin=ctx.gross_margin,
           min_margin=ctx.min_margin, weekly_sales=weekly_sales, stock_value=ctx.stock_value,
           supplier_lead_time_weeks=None if ctx.supplier_lead_time_days is None else ctx.supplier_lead_time_days / 7)
    for i, rival in enumerate(ctx.competitors):
        kb.add("Competitor", f"rival_{i}", label=rival)
        kb.relate(name, "competes_with", f"rival_{i}")
    return kb


def advise(got: IntakeResult, vitals: list[Fact], fc: Forecast | None, ctx: BusinessContext,
           extra: list[Fact] | None = None) -> tuple[list[Card], list[Recommendation], list[Fact], list[str]]:
    facts = fx.build(got, vitals, fc, ctx)
    if extra:  # marketplace mode: model-scored rival and share facts replace base rates
        names = {f.name for f in extra}
        facts = [f for f in facts if f.name not in names] + extra
    eng = engine()
    res = eng.run(facts)
    trace = res.explain() + [f"conflict: {c}" for c in res.conflicts]

    series = got.series[0]
    weekly_sales = sum(series.values[-4:]) / 4 * (1 if series.frequency == "W" else 12 / 52)
    base = {f.name: f for f in facts}
    kb = knowledge(ctx, weekly_sales)

    recs, blocked = [], dict(res.blocked)
    for move, f in res.recommendations.items():
        violations = kb.check("you", [move])
        own = [v for v in violations if v["subject"] == move]
        if own:
            blocked[move] = "; ".join(v["reason"] for v in own)
            trace.append(f"safety gate: {move} blocked by constraint(s) {[v['constraint'] for v in own]}")
            continue
        fired = [t.rule.id for t in res.trace if f"recommend:{move}" in t.conclusions]
        spec = kb.classes[move]
        recs.append(Recommendation(action=spec.get("label") or move, reason=f.value, rules_fired=fired,
                                   expected_effect=f"assumed demand x{spec.get('demand_lift')} ({spec.get('source')})"))
    why_not = {m: eng.why_not(f"recommend:{m}", facts) for m in EXPLAIN_IF_ABSENT if m not in res.recommendations}

    pulse = base.get("market_pulse")
    warnings = {k: v for k, v in res.warnings.items() if k not in DATA_NOTES}
    notes = {k: v.value for k, v in res.warnings.items() if k in DATA_NOTES}
    rival_details = {k: base[k].model_dump() for k in fx.FACTS if k.startswith("rival") and k in base}
    if "price_cut_unpredictable" in res.memory:
        rival_details["note"] = "Rival price cuts are rare (about 2% of months) and our models cannot predict them; tell us if you see one."
    cards = [
        Card(key="warnings", title="What will go wrong", source=Source.PREDICTED,
             headline=f"{len(warnings)} warning(s), {len(res.risks)} risk(s) for the next 90 days" if warnings or res.risks else "No warnings",
             details={"warnings": {k: v.value for k, v in warnings.items()}, "add_to_unlock": notes,
                      "diagnoses": {k: v.value for k, v in res.diagnoses.items()},
                      "premortem": {k: v.value for k, v in res.risks.items()},
                      "certainty": {k: v.certainty for k, v in {**warnings, **res.risks}.items()}}),
        Card(key="market", title="Your market", source=Source.MEASURED if pulse else Source.STUB,
             headline=f"Next weeks look {pulse.value} ({pulse.evidence})" if pulse else "Need 8+ weeks to read the market",
             details={k: base[k].model_dump() for k in ("market_pulse", "market_share", "share_change_expected") if k in base}),
        Card(key="rivals", title="Your competitors", source=Source.INFERRED,
             headline=(f"{int(base['rival_count'].value)} rival(s); estimates, not observations" if "rival_count" in base else "Name your competitors"),
             details=rival_details),
        Card(key="next_move", title="What to do next", source=Source.PREDICTED,
             headline=recs[0].action if recs else ("Add your finances to get a recommendation" if "need_finances" in notes else "No action needed"),
             details={"recommendations": [r.model_dump() for r in recs], "blocked": blocked, "why_not": why_not}),
    ]
    if fc is not None:
        # M9 planner: the cheapest sequence of moves that reaches the profit target, never using a move the rules forbid.
        forbidden = set(res.forbidden) | {m for m in blocked}
        p = planner.plan(series, fc, ctx, forbidden)
        if p is not None:
            cards[-1].details["plan"] = p
            trace.append(f"planner: {'found ' + '; '.join(p['steps']) if p['found'] else 'no plan reaches the target'} "
                         f"({p['nodes_expanded']} nodes expanded)")
    return cards, recs, list(res.memory.values()), trace
