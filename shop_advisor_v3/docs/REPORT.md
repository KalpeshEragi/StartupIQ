# Startup IQ: final report (CELBC710, Tasks 1 to 10)

**Startup IQ** is a weekly business checkup for small sellers. A founder uploads a sales sheet (any reasonable CSV or Excel layout) and a few numbers (cash, fixed costs, margin, stock, supplier lead time, rivals). The system returns seven cards: their data, a 12-week forecast with a range, their money, what will go wrong (warnings, diagnoses, a 90-day pre-mortem), their market, their competitors (estimates), and what to do next (rule-based recommendations plus the cheapest plan found by search). Every output says whether it was measured, predicted or inferred, and why.

Run it: `pip install -e ".[dl,api,dev]"`, then `python -m startupiq.clean` and `uvicorn startupiq.api.app:app`, and open http://127.0.0.1:8000. Technical record: `devdocs/` (architecture, evaluation protocol and log, 15 decision records, leakage checklist). Every number below has a row in `devdocs/evaluation_log.md` and a generated table in `docs/results/`.

## Architecture (Task 9)

```
upload -> M0 intake (rules)            reads messy files, detects columns, grades the data, refuses thin data
       -> M1 vitals (formulas)         growth, runway, stock cover
       -> M3 forecast (ML + DL)        hybrid: Auto's per-series choice + a TCN candidate
       -> M5 market pulse (rule)       same weeks last year
       -> M7 rivals and share (ML)     random forest moves, LightGBM share (marketplace mode); base rates otherwise
       -> facts with certainty         -> M8 rule engine (47 rules, forward + backward chaining)
       -> M10 safety gate              forbids, knowledge-model constraints, refusals
       -> M9 planner (A* search)       cheapest plan of moves reaching a profit target
       -> cards in the web app (FastAPI + one HTML page)
```

DL provides representation and sequence learning (the TCN forecaster), ML provides prediction (rivals, share, Auto), AI provides decisions (rules, constraints, search). Each technique was chosen per module and argued against its alternatives (`devdocs/decisions/0001`).

## Module 1: knowledge and decisions

**Task 1, knowledge model.** Frames (slots with default, range and if-needed procedures; inheritance Thing > Agent > Business > SmallSeller / Competitor; Move subclasses from `configs/moves.yaml`) plus a semantic network (networkx: is_a, operates_in, competes_with, affects, requires) plus a constraint checker (cash, margin floor, stock cover, runway, lead time, legality). Validated on 8 synthetic scenarios: 8 of 8 give exactly the expected violations. Figures: `docs/results/figures/16_knowledge_frames.png`, `17_knowledge_moves.png`. Decision 0007.

**Task 2, rule engine.** Own engine (decision 0003): working memory, agenda, conflict resolution by priority then specificity, refraction, MYCIN certainty factors, forward chaining, and backward chaining that explains why a move was not recommended ("needs stock cover >= 3, which is false (1.0)"). 47 rules in `configs/rules.yaml`, each with a firing and a non-firing test (47 / 47). Model outputs enter as facts whose certainty comes from measured model quality. Founder scenarios: 47 of 47 expectations met (author-written, so a consistency check, not independent validation). Two rules were fixed after the first real-data run (share-alert threshold, "hold" despite a visible problem). Decision 0013.

**Task 3, search.** State = a plan of up to 4 moves at weeks 0..10; transition = a 12-week simulator on the forecast; goal = profit target with cash never below zero; cost = cash spent plus margin given away; heuristic = fractional-knapsack bound. On 12 problems from real uploads: A* optimal in 10 of 10 solvable problems (matching uniform-cost search every time) and 5-50x fewer expansions on medium targets; BFS optimal in 3 (fewest moves, not cheapest), DFS in 2, greedy in 2. Decision 0014.

## Module 2: data and classical ML

**Task 4, data.** Four sources (decision 0002): Rossmann (1.02M store-days), UCI Online Retail II (1.07M lines), Olist (112k items, 3,095 sellers), FRED and Wikipedia signals. Scripted cleaning with before/after counts (`docs/results/cleaning_log.md`); 15 EDA figures with computed findings (`eda_summary.md`). Intake for founders' files handles title rows, ₹ amounts, no headers, Hindi headers, wide sheets, several sheets and US/Indian dates: 25 of 25 broken-file pack cases and 3 of 3 real exports correct (decision 0011).

**Task 5, features and splits.** Origin-based features (no information from after the forecast start; tests change future sales and check features do not move), temporal splits, test scored once. Leakage checklist with 20 checks (`devdocs/leakage_checklist.md`). Competitor-move labels: exit, price cut (same product, revealing that 25% "price cuts" in exploration were product-mix noise; real ones are 2%), expansion, stagnate. Feature selection did not help (Rossmann: 0.134 -> 0.163; competitor: +0.001) and is documented.

**Task 6, classical models (test results, scored once).**

| Problem | Baseline | Chosen model | Test |
| --- | --- | --- | --- |
| Rossmann daily forecast (RMSPE) | same day last year 0.160 | Random forest | **0.130** |
| Weekly forecast, 2,539 series (MASE) | moving average 0.963 | Auto (per-series pick) | **0.810** |
| Competitor moves (macro-F1) | same as last month 0.228 | Random forest | **0.366** |
| Next-month market share (MAE, pct points) | persistence 1.99 | LightGBM (L1) | **1.55** |
| Market pulse (macro-F1) | same weeks last year **0.618** | LightGBM 0.543 | model lost: the rule ships |
| Store segments | | K-means, k = 3 | silhouette 0.19, stable (ARI 0.99) |

Decision 0012.

## Module 3: deep learning

**Task 7.** MLP (basic), LSTM with attention, TCN (1D-CNN), all with non-crossing P10/P50/P90 heads and pinball loss, AdamW, gradient clipping, early stopping, random search (5 settings each), 3 seeds; trained on the RTX 2050 GPU. Validation:

| Model | Weekly pooled MASE | Weekly, sparse products | Rossmann daily RMSPE |
| --- | --- | --- | --- |
| Best classical | Auto 0.616 | Auto 0.703 | Random forest 0.119 |
| MLP | 0.720 | 0.617 | 0.145 |
| LSTM + attention | 0.716 | 0.607 | 0.159 |
| TCN | 0.686 | **0.600** | 0.131 |
| **Hybrid (Auto + TCN candidate)** | **0.603 ± 0.002** | 0.682 | |

No deep model beats the best classical model alone; on sparse products every deep model does. The hybrid lets Auto pick the TCN per series by backtest and meets the decision rule (beats Auto by more than one sd, coverage 0.75), so it serves. Attention lowers LSTM error from 0.776 to 0.695; dropout 0.3 lifts coverage from 0.74 to 0.83. A bug worth knowing: Rossmann training origins every 14 days all fell on a Tuesday, the model learned the weekday from position, and RMSPE was 0.549 until origins were spaced 10 days apart (0.145). Decision 0015.

**Task 8, interpretability** (`docs/results/dl_interpret.md`). Gradient x input saliency: the TCN puts 29% of its attribution on the last 4 weeks (43% for stores), and 16% on "the same week last year". LSTM attention does **not** agree with occlusion (median Spearman -0.11, top-5 overlap 4%), while gradient saliency does (0.63): attention weights are not a faithful explanation here, which is why the product explains forecasts with backtest errors and saliency, not attention. TCN first-layer filters mostly fire on a sudden spike in the latest week. Figures 18-21.

## Module 4: integration and validation

**Task 9.** `pipeline.analyze()` wires every module; FastAPI app with one HTML page (`src/startupiq/api/`), checked end to end in a browser and by tests. 245+ automated tests.

**Task 10, validation on unseen businesses** (`docs/results/validation.md`). 300 Olist sellers never used for training or model choice, each uploaded like a founder's file with the last 12 weeks hidden: product forecaster MASE 0.982 vs naive 1.036 vs moving average 0.936; P10-P90 coverage 0.56. Better than both baselines for small sellers (0.917), worse for large ones (1.105). Full analysis takes 0.26 s (median) per upload; identical uploads give identical reports (40 of 40). Ethics, fairness and licences: `docs/ETHICS.md`.

## Honest limits

1. On unseen marketplace sellers the forecaster does not beat a plain moving average, and its ranges are too narrow (56% coverage).
2. Competitor and share outputs are inferences; rival price cuts cannot be predicted.
3. Move effects in the planner are assumptions; only the promotion lift is grounded in data.
4. No real Indian small business has used it yet; a 3-5 business pilot is the next step.
5. Data licences allow academic use only.
