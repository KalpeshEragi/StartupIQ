"""Load rules from configs/rules.yaml.

Format of one rule:
    - id: R20
      name: Stock runs out before the supplier can deliver
      group: stock
      priority: 90          # higher fires first
      cf: 0.9               # how sure the rule's author is that the conclusion follows
      when:                 # all must hold; value "$other_fact" compares two facts
        - [stock_cover_weeks, "<", "$supplier_lead_time_weeks"]
      then:
        - assert: {stockout_risk: true}
        - warn: stockout
          text: "Stock lasts {stock_cover_weeks:.1f} weeks; your supplier needs {supplier_lead_time_weeks} weeks."
      fires_when: {stock_cover_weeks: 1.0, supplier_lead_time_weeks: 2}   # test case: rule must fire
      not_when:   {stock_cover_weeks: 5.0, supplier_lead_time_weeks: 2}   # test case: rule must not fire
"""

from __future__ import annotations

from pathlib import Path

import yaml

from startupiq.rules.engine import ACTIONS, Conclusion, Condition, Engine, OPS, Rule

RULES_FILE = Path(__file__).resolve().parents[3] / "configs" / "rules.yaml"
UNARY = ("exists", "missing")


def _condition(spec) -> Condition:
    if len(spec) == 2 and spec[1] in UNARY:
        return Condition(spec[0], spec[1])
    fact, op, value = spec
    if op not in OPS and op not in ("in", "between"):
        raise ValueError(f"unknown operator {op!r}")
    return Condition(fact, op, value)


def _conclusions(items) -> list[Conclusion]:
    out = []
    for item in items:
        kind = next(k for k in item if k in ACTIONS)
        if kind == "assert":
            for name, value in item["assert"].items():
                out.append(Conclusion("assert", name, value))
        else:
            out.append(Conclusion(kind, item[kind], True, item.get("text", "")))
    return out


def load(path: Path = RULES_FILE) -> list[Rule]:
    rules = []
    for spec in yaml.safe_load(path.read_text(encoding="utf-8")):
        rules.append(Rule(id=spec["id"], name=spec["name"], group=spec["group"], priority=spec.get("priority", 50),
                          cf=spec.get("cf", 1.0), when=[_condition(c) for c in spec.get("when", [])],
                          then=_conclusions(spec["then"]), fires_when=spec.get("fires_when", {}), not_when=spec.get("not_when", {})))
    return rules


def engine(path: Path = RULES_FILE) -> Engine:
    return Engine(load(path))
