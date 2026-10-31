# 0007: Knowledge representation

- Status: Accepted (2026-10-24): implemented in `src/startupiq/knowledge/`; 8 of 8 validation scenarios pass
- Date: 2026-10-24
- Module / task: Task 1

## Problem

Represent businesses, products, categories, competitors, moves, metrics and constraints so that rules and search can use them and invalid states are rejected.

## Options considered

| Option | For | Against |
| --- | --- | --- |
| First-order logic | Precise, supports proofs | Awkward for attributes with defaults; hard to explain to users |
| **Frames** | Entities with slots, defaults and inheritance (a Startup is-a Business); maps directly to Python dataclasses | No built-in relation reasoning |
| **Semantic network** | Relations as a graph (competes_with, affects, requires); easy to draw and query with networkx | Weak on attribute constraints |
| Ontology (OWL, owlready2) | Standard, reasoner can check consistency | Heavy tooling for a small domain; slower to iterate |

## Decision

Frames for entities and their attributes, plus a semantic network for relations, plus a constraint checker in Python. This combination covers attributes, inheritance, relations and constraints with little tooling, and the network diagram is a clear Task 1 deliverable. OWL export is optional if time allows.

## Validation plan

Eight labelled synthetic scenarios: four valid, four that each break one constraint (move cost over budget, price cut below margin floor, expansion without stock cover, price-coordination move). The checker must accept and reject exactly those.

## Implementation (Step 6)

- **Frames** (`knowledge/frames.py`): slots with value, default, range and an if-needed procedure. Lookup order: own value, own if-needed, then up the is_a chain, then the nearest default. `get(slot, explain=True)` says where a value came from ("computed by if-needed on Business", "default from Competitor"), which the explanations reuse.
- **Class frames** (`knowledge/ontology.py`): Thing > Agent > Business > SmallSeller / Competitor; Product; Category; Metric; Constraint; Move > one frame per move in `configs/moves.yaml`. Derived slots are if-needed procedures: monthly sales, runway, stock cover. A Competitor inherits `observed = False` and base-rate move probabilities: the representation itself records that a founder cannot see a rival's books.
- **Semantic network** (`knowledge/kb.py`, networkx): is_a, instance_of, operates_in, sells, belongs_to, competes_with, affects (with sign), requires. Queries: related nodes, moves affecting a metric, shortest explanation path.
- **Constraints** (`knowledge/constraints.py`): cash covers cost, margin above floor, lead time known, runway of 6+ months for slow-payback moves, stock covers a promotion or a new channel, legality, rivals in the same category, slot ranges.
- **Figures**: `docs/results/figures/16_knowledge_frames.png`, `17_knowledge_moves.png`; full listing in `docs/results/knowledge.md` (`python -m startupiq.knowledge`).

## Validation

`tests/fixtures/knowledge/scenarios.yaml`: 4 consistent businesses and 4 that each break specific constraints; the checker reports exactly the expected violations in all 8 (`tests/test_knowledge.py`). Writing the scenarios exposed two mistakes in the scenario data, not the checker: the "healthy" cafe really had only 2.7 weeks of stock (a new channel needs 3), and the "cash-short" case also broke the cash constraint. Both were corrected so each invalid case breaks the constraint it is meant to.
