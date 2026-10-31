# 0003: Write our own rule engine

- Status: Accepted; implemented in `src/startupiq/rules/engine.py` (Step 7), rule base in decision 0013
- Date: 2026-10-24
- Module / task: M8, M10; Task 2

## Problem

Task 2 needs IF-THEN rules with forward or backward chaining and visible rule firing. Rules must consume `Fact`s that carry a certainty from ML models.

## Options considered

| Option | For | Against |
| --- | --- | --- |
| experta (pyknow fork) | Rete-based, CLIPS-like syntax | Unmaintained since about 2020; known breakage on recent Python (`frozendict` dependency); we run Python 3.13 |
| durable_rules | Fast, maintained on and off | Rules live in its own DSL; tracing and certainty factors are awkward; heavier install |
| CLIPS via clipspy | The textbook expert-system shell | Separate language to learn; harder to pass Python facts and certainties both ways |
| Own engine, about 150 lines | Full control of trace, certainty factors, backward chaining; rules in readable YAML; easy to test; shows we understand inference | No Rete optimisation (irrelevant at 40 rules) |

## Criteria

Works on Python 3.13; readable trace for the UI; supports both chaining directions; supports certainty factors; easy unit tests.

## Decision

Own engine. Working memory of facts, agenda of matched rules, conflict resolution by priority then specificity (more conditions first), refraction (a rule fires once per fact set), certainty combined MYCIN-style (rule CF times the minimum CF of its conditions). Backward chaining for "why not X?" walks from a goal to unmet conditions.

## Consequences

We own the bugs, so every rule gets a firing and a non-firing test. Performance is not a concern at this size.
