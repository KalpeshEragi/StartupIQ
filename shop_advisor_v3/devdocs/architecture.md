# Architecture

## Flow

```
upload (CSV/Excel) + BusinessContext
  -> M0 intake ............ IntakeResult (tier, weekly SalesSeries, cleaning log)
  -> M1 vitals ............ Facts (growth, runway, stock cover)            deterministic
  -> M2 anomalies ......... Facts                                          ML
  -> M3 forecast .......... Forecast (P10/P50/P90)                         DL / ML
  -> M4 reviews ........... Facts (sentiment, themes)                      DL (stretch)
  -> M5 market pulse ...... Facts (trend regime)                           ML
  -> M6 peer cohort ....... Facts (segment, peer growth)                   ML
  -> M7 rival radar ....... Facts (move probabilities, share), INFERRED    ML
  -> M8 diagnosis ......... Recommendations + rule trace                   rules
  -> M9 planner ........... plan of moves, simulated with M3/M7            search
  -> M10 safety gate ...... drops or rewrites unsafe or unsupported output rules
  -> M11 note ............. Cards for the UI                               templates
```

Everything after M0 talks through the types in [`src/startupiq/contracts.py`](../src/startupiq/contracts.py). The single entry point is `startupiq.pipeline.analyze(table, ctx) -> Report`.

## Intake (M0) inside

```
read_file / read_upload -> reader.py (encodings, delimiter, header row, sheets, Total rows)
  -> layout.py (periods across / products across -> long)
  -> columns.py (role per column: name score x content score -> confidence)
  -> series.py (amount rule, dates, duplicates, returns, partial weeks, gaps, outliers, tier)
  -> IntakeResult (tier, series, column_map, confidence, needs_confirmation, messages, cleaning_log)
```

Entry points: `pipeline.analyze_file(path)`, `pipeline.analyze(df, column_map=...)` (the UI passes the user's corrections as `column_map`).

## Contracts in one paragraph

`IntakeResult` carries the data tier and cleaned series. Every module that produces knowledge emits `Fact`s with a `certainty` (0 to 1) and a `Source` (measured, predicted, inferred, stub). The rule engine reads only facts, so it never depends on which model produced them. The UI renders `Card`s; a card the tier does not allow carries `locked_reason` instead of a number.

## Where each task lives

| Task | Code | Status |
| --- | --- | --- |
| 1 Knowledge model | `src/startupiq/knowledge/`, `configs/moves.yaml` | Done (Step 6): frames, semantic network, constraint checker, 8 validation scenarios, figures 16-17 |
| 2 Rule engine | `src/startupiq/rules/`, `configs/rules.yaml`, `src/startupiq/advisor.py` | Done (Step 7): 47 rules, forward and backward chaining, certainty factors, safety gate; fills the warnings, market, rivals and next-move cards |
| 3 Search | `src/startupiq/search/` | Done (Step 10): plan state space, simulator on the forecast, BFS/DFS/UCS/A*/greedy, comparison on 12 problems; A* plan on the next-move card |
| 4 Load, clean, EDA | `src/startupiq/clean/`, `src/startupiq/eda.py` | Done (Steps 1-2): 4 cleaners with logs, 15 figures, data cards |
| 5 Features, splits | `src/startupiq/features/`, `src/startupiq/intake/` | Done (Steps 3-4): intake; origin-based features for Rossmann, the weekly panel and Olist; competitor-move labels; splits in `features/splits.py`; leakage checklist with tests |
| 6 Classical ML | `src/startupiq/models/` | Done (Step 5): six experiments (`models/run.py`), choices in decision 0012; the forecast card uses the validated Auto method |
| 7 Deep learning | `src/startupiq/dl/` | Done (Step 8): MLP, LSTM+attention, TCN; hybrid serves the forecast card (decision 0015) |
| 8 Interpretability | `src/startupiq/dl/interpret.py` | Done (Step 9): saliency, attention vs occlusion, TCN filters, comparison table |
| 9 Integration | `src/startupiq/pipeline.py`, `src/startupiq/advisor.py`, `src/startupiq/api/` | Done (Step 11): pipeline + FastAPI app with one HTML page; `uvicorn startupiq.api.app:app` |
| 10 Validation, ethics | `src/startupiq/evaluate/validation.py`, `docs/ETHICS.md`, `docs/REPORT.md`, `Dockerfile` | Done (Step 12): unseen sellers, fairness by size, latency, determinism; ethics and licences; report and slide outline |

## Stub policy

A module that is not built returns a locked card with `Source.STUB`. A stub must never show a number that looks validated. Since Step 5 the forecast card uses the validated Auto method (`models/forecast_baseline.py` -> `models/forecast_methods.py`) and is marked PREDICTED; since Step 7 the warnings, market, rivals and next-move cards come from the rule engine (`advisor.py`); no card is a stub any more.

## Advisor (M8 + M10) inside

```
intake + vitals + forecast (+ marketplace facts from the Step 5 models)
  -> rules/facts.py      facts with value, certainty, source (decision 0013)
  -> rules/engine.py     forward chaining -> warnings, diagnoses, recommendations, forbids, pre-mortem risks
  -> knowledge/kb.py     each recommended move checked against constraints (safety gate)
  -> engine.why_not      backward chaining: why common moves were not recommended
  -> search/plan.py      A*: cheapest plan of moves reaching the profit target, excluding forbidden moves
  -> cards: warnings, market, rivals, next_move (+ the full trace in Report.trace)
```

Marketplace mode: `rules.facts.marketplace(seller_id, category, month)` scores a seller's rivals with the competitor random forest and its next-month share with the share LightGBM (`models/serve.py`, cached in `data/models/`), and passes them to `pipeline.analyze(..., extra_facts=...)`.
