"""Task 2: rule engine semantics, every rule's firing and non-firing case, and the founder scenarios."""

from pathlib import Path

import pytest
import yaml

from startupiq.contracts import Fact
from startupiq.rules import loader
from startupiq.rules.engine import Conclusion, Condition, Engine, Rule

RULES = loader.load()
ENGINE = Engine(RULES)
SCENARIOS = yaml.safe_load((Path(__file__).parent / "fixtures" / "rules" / "scenarios.yaml").read_text(encoding="utf-8"))


def facts(d: dict) -> list[Fact]:
    return [Fact(name=k, value=v) for k, v in d.items()]


def fired(result) -> set[str]:
    return {f.rule.id for f in result.trace}


# ---------- every rule has a firing and a non-firing case ----------

@pytest.mark.parametrize("rule", RULES, ids=[r.id for r in RULES])
def test_rule_fires_on_its_example(rule):
    assert rule.id in fired(ENGINE.run(facts(rule.fires_when)))


@pytest.mark.parametrize("rule", [r for r in RULES if r.not_when is not None], ids=[r.id for r in RULES if r.not_when is not None])
def test_rule_does_not_fire_on_its_counter_example(rule):
    assert rule.id not in fired(ENGINE.run(facts(rule.not_when)))


def test_every_rule_has_examples_and_unique_ids():
    assert len({r.id for r in RULES}) == len(RULES) >= 30
    assert all(r.fires_when is not None for r in RULES)


# ---------- engine semantics ----------

def mini(*rules) -> Engine:
    return Engine(list(rules))


def test_certainty_is_rule_cf_times_weakest_condition():
    e = mini(Rule("A", "a", "t", [Condition("x", ">", 1), Condition("y", "==", True)], [Conclusion("assert", "z", True)], cf=0.8))
    r = e.run([Fact(name="x", value=5, certainty=0.9), Fact(name="y", value=True, certainty=0.5)])
    assert r.memory["z"].certainty == pytest.approx(0.4)


def test_two_rules_for_the_same_conclusion_combine_like_mycin():
    e = mini(Rule("A", "a", "t", [Condition("x", ">", 1)], [Conclusion("assert", "z", True)], cf=0.6),
             Rule("B", "b", "t", [Condition("y", ">", 1)], [Conclusion("assert", "z", True)], cf=0.5))
    r = e.run(facts({"x": 2, "y": 2}))
    assert r.memory["z"].certainty == pytest.approx(0.6 + 0.5 * (1 - 0.6))


def test_weak_evidence_below_threshold_does_not_fire():
    e = mini(Rule("A", "a", "t", [Condition("x", ">", 1)], [Conclusion("assert", "z", True)]))
    assert "z" not in e.run([Fact(name="x", value=5, certainty=0.1)]).memory


def test_conflict_resolution_priority_then_specificity():
    e = mini(Rule("LOW", "l", "t", [Condition("x", ">", 0)], [Conclusion("assert", "a", 1)], priority=10),
             Rule("HIGH", "h", "t", [Condition("x", ">", 0)], [Conclusion("assert", "b", 1)], priority=90),
             Rule("SPEC", "s", "t", [Condition("x", ">", 0), Condition("y", ">", 0)], [Conclusion("assert", "c", 1)], priority=10))
    order = [f.rule.id for f in e.run(facts({"x": 1, "y": 1})).trace]
    assert order == ["HIGH", "SPEC", "LOW"]


def test_forward_chaining_reaches_derived_facts():
    r = ENGINE.run(facts({"runway_months": 2.0}))
    order = [f.rule.id for f in r.trace]
    assert order.index("R10") < order.index("R13")  # cash_risk is derived before it is used
    assert "new_product" in r.forbidden


def test_forbid_beats_recommend():
    r = ENGINE.run(facts({"history_weeks": 12, "runway_months": 14.0, "forecast_growth": 0.2, "stock_cover_weeks": 6.0, "band_width": 0.3}))
    assert "channel_expansion" in r.blocked and "channel_expansion" not in r.recommendations


def test_comparison_with_another_fact_and_type_mismatch():
    e = mini(Rule("A", "a", "t", [Condition("x", "<", "$y")], [Conclusion("assert", "z", True)]))
    assert "z" in e.run(facts({"x": 1, "y": 2})).memory
    assert "z" not in e.run(facts({"x": "profitable", "y": 2})).memory  # text vs number is false, not an error


def test_wildcard_missing():
    r = ENGINE.run(facts({"history_weeks": 60, "runway_months": 999.0}))
    assert "hold" in r.recommendations  # fires only because no recommend:* fact exists
    r = ENGINE.run(facts({"history_weeks": 60, "runway_months": 2.0}))
    assert "hold" not in r.recommendations and "cost_reduction" in r.recommendations


def test_templates_fill_values():
    r = ENGINE.run(facts({"runway_months": 2.04}))
    assert "2.0 months" in r.warnings["runway_critical"].value


def test_backward_chaining_proves_and_explains():
    base = facts({"runway_months": 2.0})
    proof = ENGINE.prove("forbid:new_product", base)
    assert proof["status"] == "proved" and proof["by"]["rule"].startswith("R13")
    reasons = ENGINE.why_not("recommend:channel_expansion", facts({"runway_months": 14.0, "forecast_growth": 0.2, "stock_cover_weeks": 1.0}))
    assert "stock_cover_weeks >= 3, which is false (1.0)" in reasons[0]


def test_why_not_marks_missing_information():
    reasons = ENGINE.why_not("recommend:restock", facts({"history_weeks": 60}))
    assert any("information not given" in r for r in reasons)


# ---------- founder scenarios ----------

def check(result, spec) -> list[str]:
    problems = []
    for m in spec.get("recommend", []):
        if m not in result.recommendations:
            problems.append(f"expected recommend {m}")
    for m in spec.get("not_recommend", []):
        if m in result.recommendations:
            problems.append(f"unexpected recommend {m}")
    for kind, have in (("warn", result.warnings), ("forbid", result.forbidden), ("diagnose", result.diagnoses), ("risk", result.risks)):
        for k in spec.get(kind, []):
            if k not in have:
                problems.append(f"expected {kind} {k}")
    return problems


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_founder_scenario(name):
    spec = SCENARIOS[name]
    result = ENGINE.run(facts(spec["facts"]))
    assert check(result, spec) == [], result.explain()
