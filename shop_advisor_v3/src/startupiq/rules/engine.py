"""Rule engine with forward and backward chaining and certainty factors (Task 2, decision 0003).

Working memory holds `Fact`s (name, value, certainty, source, evidence). A rule fires when
all its conditions hold with certainty >= CF_THRESHOLD. Its conclusions get
certainty = rule.cf x min(condition certainties) (MYCIN). Two rules concluding the same
fact with the same value combine as a + b(1 - a).

Forward chaining (`run`): data-driven. Repeatedly build the agenda (matched, not yet fired
rules), pick one by conflict resolution (priority, then specificity = number of conditions,
then order in the file), fire it, until nothing new can fire.

Backward chaining (`prove`, `why_not`): goal-driven. To prove a fact, find rules that
conclude it and try to prove their conditions from the given facts, recursively; report
which conditions failed or are unknown (missing data).

Conclusions are facts with reserved prefixes: `warn:<key>`, `diagnose:<key>`, `recommend:<move>`,
`forbid:<move>`, `risk:<key>` (pre-mortem). A forbid always beats a recommend. Several diagnoses
can hold at once (a shop can be both growing and cash-starved), so each is its own fact.
"""

from __future__ import annotations

import operator
import re
import string
from dataclasses import dataclass, field
from typing import Any

from startupiq.contracts import Fact, Source

CF_THRESHOLD = 0.2  # as in MYCIN: weaker evidence is treated as unknown
MAX_CYCLES = 500
OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq, "!=": operator.ne}
ACTIONS = ("assert", "warn", "diagnose", "recommend", "forbid", "risk")


@dataclass
class Condition:
    fact: str
    op: str
    value: Any = None

    def __str__(self) -> str:
        return f"{self.fact} {self.op} {self.value}" if self.op not in ("exists", "missing") else f"{self.fact} {self.op}"


@dataclass
class Conclusion:
    kind: str  # assert | warn | diagnose | recommend | forbid | risk
    name: str  # fact name (assert) or key/move
    value: Any = True
    text: str = ""

    @property
    def fact_name(self) -> str:
        return self.name if self.kind == "assert" else f"{self.kind}:{self.name}"


@dataclass
class Rule:
    id: str
    name: str
    group: str
    when: list[Condition]
    then: list[Conclusion]
    priority: int = 50
    cf: float = 1.0
    fires_when: dict = field(default_factory=dict)
    not_when: dict = field(default_factory=dict)

    @property
    def specificity(self) -> int:
        return len(self.when)

    def concludes(self, fact_name: str) -> bool:
        return any(c.fact_name == fact_name for c in self.then)


@dataclass
class Firing:
    cycle: int
    rule: Rule
    matched: dict[str, Any]
    certainty: float
    conclusions: list[str]


class _SafeDict(dict):
    def __missing__(self, key):
        return "?"


def render(text: str, wm: dict[str, Fact]) -> str:
    """Fill {fact} or {fact:.1f} from working memory; unknown facts show as '?'."""
    values = _SafeDict({k: f.value for k, f in wm.items() if ":" not in k})
    try:
        return string.Formatter().vformat(text, (), values)
    except (ValueError, TypeError):
        return re.sub(r"\{([^}:]+)(:[^}]*)?\}", lambda m: str(values.get(m.group(1), "?")), text)


class Engine:
    def __init__(self, rules: list[Rule]):
        ids = [r.id for r in rules]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate rule ids")
        self.rules = rules
        self.order = {r.id: i for i, r in enumerate(rules)}

    # ---------- matching ----------
    @staticmethod
    def _lookup(wm: dict[str, Fact], name: str) -> list[Fact]:
        if name.endswith("*"):
            return [f for k, f in wm.items() if k.startswith(name[:-1]) and f.certainty >= CF_THRESHOLD]
        f = wm.get(name)
        return [f] if f is not None and f.certainty >= CF_THRESHOLD else []

    def _test(self, cond: Condition, wm: dict[str, Fact]) -> tuple[bool, float, Any]:
        """(holds, certainty of the evidence, value seen)."""
        found = self._lookup(wm, cond.fact)
        if cond.op == "missing":
            return (not found, 1.0, None)
        if cond.op == "exists":
            return (bool(found), max((f.certainty for f in found), default=0.0), found[0].value if found else None)
        if not found:
            return (False, 0.0, None)
        fact = found[0]
        target, target_cf = cond.value, 1.0
        if isinstance(target, str) and target.startswith("$"):
            other = self._lookup(wm, target[1:])
            if not other:
                return (False, 0.0, fact.value)
            target, target_cf = other[0].value, other[0].certainty
        try:
            if cond.op == "in":
                ok = fact.value in target
            elif cond.op == "between":
                ok = target[0] <= fact.value < target[1]
            else:
                ok = OPS[cond.op](fact.value, target)
        except TypeError:
            ok = False  # comparing a number with text (e.g. runway "profitable") is simply false
        return (bool(ok), min(fact.certainty, target_cf), fact.value)

    def match(self, rule: Rule, wm: dict[str, Fact]) -> tuple[bool, float, dict]:
        cf, seen = 1.0, {}
        for cond in rule.when:
            ok, c, v = self._test(cond, wm)
            if not ok:
                return False, 0.0, {}
            cf = min(cf, c)
            seen[cond.fact] = v
        return True, cf, seen

    # ---------- forward chaining ----------
    def run(self, facts: list[Fact]) -> "Result":
        wm: dict[str, Fact] = {f.name: f.model_copy() for f in facts}
        fired: set[str] = set()
        trace: list[Firing] = []
        conflicts: list[str] = []
        for cycle in range(1, MAX_CYCLES + 1):
            agenda = []
            for r in self.rules:
                if r.id in fired:
                    continue  # refraction: a rule fires at most once per run
                ok, cf, seen = self.match(r, wm)
                if ok:
                    agenda.append((r, cf, seen))
            if not agenda:
                break
            agenda.sort(key=lambda x: (-x[0].priority, -x[0].specificity, self.order[x[0].id]))
            rule, cond_cf, seen = agenda[0]
            fired.add(rule.id)
            cf = round(rule.cf * cond_cf, 4)
            names = []
            for c in rule.then:
                name = c.fact_name
                value = c.value if c.kind == "assert" else render(c.text, wm)
                evidence = f"{rule.id} {rule.name}: " + ", ".join(f"{k}={_fmt(v)}" for k, v in seen.items())
                old = wm.get(name)
                if old is None or old.certainty < CF_THRESHOLD:
                    wm[name] = Fact(name=name, value=value, certainty=cf, source=Source.INFERRED if c.kind == "assert" else Source.MEASURED,
                                    evidence=evidence)
                elif old.value == value or c.kind != "assert":
                    old.certainty = round(old.certainty + cf * (1 - old.certainty), 4)  # MYCIN combination
                    old.evidence += f" | {evidence}"
                else:
                    conflicts.append(f"{rule.id} wanted {name}={value!r} (cf {cf}) but {name}={old.value!r} (cf {old.certainty}) is held")
                    if cf > old.certainty:
                        wm[name] = Fact(name=name, value=value, certainty=cf, source=Source.INFERRED, evidence=evidence)
                names.append(name)
            trace.append(Firing(cycle, rule, seen, cf, names))
        return Result(wm, trace, conflicts, self)

    # ---------- backward chaining ----------
    def prove(self, goal: str, facts: list[Fact], depth: int = 0, seen: frozenset = frozenset()) -> dict:
        """Proof tree for `goal` from the given (input) facts only."""
        wm = {f.name: f for f in facts}
        if goal in wm and wm[goal].certainty >= CF_THRESHOLD:
            return {"goal": goal, "status": "given", "value": wm[goal].value}
        candidates = [r for r in self.rules if r.concludes(goal)]
        if not candidates or depth > 8 or goal in seen:
            return {"goal": goal, "status": "unknown", "reason": "no fact and no rule concludes it"}
        attempts = []
        for r in sorted(candidates, key=lambda r: (-r.priority, self.order[r.id])):
            conds = []
            derived = dict(wm)
            for cond in r.when:
                ok, _, v = self._test(cond, derived)
                if ok:
                    conds.append({"condition": str(cond), "status": "holds", "value": _fmt(v)})
                    continue
                if cond.fact.endswith("*") or cond.op == "missing" or cond.fact in derived:
                    conds.append({"condition": str(cond), "status": "fails", "value": _fmt(v)})
                    continue
                sub = self.prove(cond.fact, list(derived.values()), depth + 1, seen | {goal})
                if sub["status"] in ("given", "proved"):
                    derived[cond.fact] = Fact(name=cond.fact, value=sub.get("value"), certainty=1.0)
                    ok, _, v = self._test(cond, derived)
                    conds.append({"condition": str(cond), "status": "holds" if ok else "fails", "value": _fmt(v), "via": sub})
                elif sub["status"] == "not proved":  # derivable in principle, but its own conditions fail
                    conds.append({"condition": str(cond), "status": "not derived", "via": sub})
                else:  # no fact and no rule: the founder has not given this information
                    conds.append({"condition": str(cond), "status": "unknown", "via": sub})
            attempt = {"rule": f"{r.id} {r.name}", "conditions": conds}
            if all(c["status"] == "holds" for c in conds):
                concl = next(c for c in r.then if c.fact_name == goal)
                return {"goal": goal, "status": "proved", "value": concl.value if concl.kind == "assert" else True, "by": attempt}
            attempts.append(attempt)
        return {"goal": goal, "status": "not proved", "attempts": attempts}

    def why_not(self, goal: str, facts: list[Fact]) -> list[str]:
        """Plain-language reasons a conclusion was not reached (backward chaining)."""
        proof = self.prove(goal, facts)
        if proof["status"] in ("given", "proved"):
            return [f"{goal} does follow: {proof.get('by', {}).get('rule', 'given')}"]
        if proof["status"] == "unknown":
            return [f"No rule can conclude {goal}."]
        return [f"{a['rule']} needs " + "; ".join(_why(c) for c in a["conditions"] if c["status"] != "holds") for a in proof["attempts"]]


@dataclass
class Result:
    memory: dict[str, Fact]
    trace: list[Firing]
    conflicts: list[str]
    engine: Engine

    def _by_kind(self, kind: str) -> dict[str, Fact]:
        return {k.split(":", 1)[1]: f for k, f in self.memory.items() if k.startswith(kind + ":") and f.certainty >= CF_THRESHOLD}

    @property
    def warnings(self) -> dict[str, Fact]:
        return self._by_kind("warn")

    @property
    def diagnoses(self) -> dict[str, Fact]:
        return self._by_kind("diagnose")

    @property
    def risks(self) -> dict[str, Fact]:
        return self._by_kind("risk")

    @property
    def forbidden(self) -> dict[str, Fact]:
        return self._by_kind("forbid")

    @property
    def recommendations(self) -> dict[str, Fact]:
        """Recommended moves that no rule forbids, strongest first."""
        rec = {m: f for m, f in self._by_kind("recommend").items() if m not in self.forbidden}
        return dict(sorted(rec.items(), key=lambda kv: -kv[1].certainty))

    @property
    def blocked(self) -> dict[str, str]:
        return {m: self.forbidden[m].value for m in self._by_kind("recommend") if m in self.forbidden}

    def explain(self) -> list[str]:
        return [f"[{f.cycle}] {f.rule.id} {f.rule.name} (cf {f.certainty:.2f}) because "
                + ", ".join(f"{k}={_fmt(v)}" for k, v in f.matched.items()) + " -> " + ", ".join(f.conclusions) for f in self.trace]


def _why(c: dict, depth: int = 0) -> str:
    """One failed condition in words, following derived sub-goals one level down."""
    if c["status"] == "unknown":
        return f"{c['condition']}, which is unknown (information not given)"
    if c["status"] == "not derived" and depth < 1:
        inner = [f"{a['rule']}: " + ", ".join(_why(x, depth + 1) for x in a["conditions"] if x["status"] != "holds")
                 for a in c["via"].get("attempts", [])]
        return f"{c['condition']}, which does not follow ({'; '.join(inner)})"
    if c["status"] == "not derived":
        return f"{c['condition']}, which does not follow"
    return f"{c['condition']}, which is false ({c.get('value')})"


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.1f}" if v.is_integer() else f"{v:.3g}"
    return str(v)
