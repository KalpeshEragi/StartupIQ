# devdocs: technical record of Startup IQ

`docs/` is for readers of the project (plan, data cards, report). `devdocs/` is for the people building it: how it works, how we measure it, and **why each technique was picked over its alternatives**. Examiners asking "why LightGBM and not random forest?" or "why an LSTM?" should find the answer here, with numbers.

| File | What it holds |
| --- | --- |
| [architecture.md](architecture.md) | Modules M0 to M11, data flow, contracts, where each syllabus task lives |
| [datasets.md](datasets.md) | Every dataset we evaluated, measured facts, licence, known problems |
| [evaluation_protocol.md](evaluation_protocol.md) | Splits, metrics, baselines, seeds, the "score test once" rule |
| [evaluation_log.md](evaluation_log.md) | Append-only log of every measured result, newest at the bottom |
| [leakage_checklist.md](leakage_checklist.md) | Every leakage risk in features, labels and splits, with the test that guards it |
| [decisions/](decisions/) | One decision record per choice: options considered, criteria, evidence, verdict |
| [templates/](templates/) | Templates for decision records and model comparisons |

## Rules for keeping this up to date

1. **Any choice between techniques gets a decision record** in `decisions/`, numbered in order. Copy `templates/decision.md`.
2. A decision starts as **Proposed** (reasoning only) and becomes **Accepted** only after the evidence is in `evaluation_log.md`. If results contradict it, mark it **Superseded** and write a new one; never edit history away.
3. **Every number reported anywhere** (report, slides, UI) must have a row in `evaluation_log.md` with the date, data, split, command that produced it.
4. Decision rules for model selection are written **before** running the experiment (see 0005). This prevents choosing the metric after seeing results.
5. Keep entries short. A table and three sentences beat a page of prose.

## Decision index

| # | Decision | Status |
| --- | --- | --- |
| [0001](decisions/0001-technique-per-module.md) | Which technique each module uses (rules vs ML vs DL vs search) | Accepted (design) |
| [0002](decisions/0002-datasets-and-licences.md) | Datasets: Rossmann, UCI Online Retail II, Olist, FRED, Wikipedia | Accepted |
| [0003](decisions/0003-rule-engine.md) | Write our own rule engine instead of experta / durable_rules / CLIPS | Accepted |
| [0004](decisions/0004-data-tiers.md) | Data-quality tiers and refusal thresholds | Accepted |
| [0005](decisions/0005-forecaster.md) | Forecast models: LSTM and 1D-CNN vs classical vs baselines | Proposed (classical bar set in 0012; DL after Step 8) |
| [0006](decisions/0006-competitor-classifier.md) | Competitor-move classifier candidates and metric | Accepted |
| [0007](decisions/0007-knowledge-representation.md) | Frames + semantic network for the knowledge model | Accepted |
| [0008](decisions/0008-search-algorithms.md) | BFS, DFS, A*, greedy and the heuristic | Accepted (see 0014) |
| [0009](decisions/0009-metrics.md) | RMSPE, MASE, coverage, macro-F1 as primary metrics | Accepted |
| [0010](decisions/0010-cleaning-rules.md) | Cleaning rules: flag before drop, never fill unrecorded gaps, impute inside training only | Accepted |
| [0011](decisions/0011-intake-design.md) | Intake: rules on names and content with confidence and questions, not a learned or LLM column mapper | Accepted |
| [0012](decisions/0012-step5-model-choices.md) | Step 5 classical choices: random forest (Rossmann), Auto (weekly product forecast), random forest (competitor moves), LightGBM L1 (share), rule (market pulse), K-means k=3 (cohorts) | Accepted |
| [0013](decisions/0013-rule-base.md) | Rule base: facts with measured certainties, 47 rules, forward/backward chaining, three-layer safety gate | Accepted |
| [0014](decisions/0014-planner.md) | Planner: plans of up to 4 moves, simulator on the forecast, cost = what is given up, knapsack heuristic; A* optimal in 12 of 12 | Accepted |
| [0015](decisions/0015-deep-forecasters.md) | Deep forecasters: MLP, LSTM+attention, TCN; the hybrid Auto + TCN serves; the weekday bug | Accepted |

## Generated outputs (rebuild, do not edit)

| Command | Writes |
| --- | --- |
| `python -m startupiq.clean` | `data/processed/*.parquet`, per-dataset JSON logs, `docs/results/cleaning_log.md` |
| `python -m startupiq.eda` | `docs/results/figures/*.png` (15 figures), `docs/results/eda_summary.md` |
| `python -m startupiq.models.run <experiment>` | Validation results: `docs/results/<experiment>.md` and `.json` (forecast_rossmann, forecast_weekly, competitor, share, clustering, regime) |
| `python -m startupiq.knowledge` | Knowledge model figures 16-17 and `docs/results/knowledge.md` |
| `python -m startupiq.models.run dl_weekly / dl_rossmann / dl_hybrid [--final]` | Deep-model results (GPU recommended) |
| `python -m startupiq.dl.interpret` | `docs/results/dl_interpret.md`, figures 18-21 |
| `python -m startupiq.evaluate.validation` | `docs/results/validation.md`: unseen sellers, fairness by size, latency |
| `python -m startupiq.search` | `docs/results/planner.md`: five algorithms on 12 planning problems |
| `python -m startupiq.rules.evaluate` | `docs/results/rules.md`: scenario agreement, coverage, behaviour on 80 real uploads |
| `python -m startupiq.models.run <experiment> --final` | Test results, scored once after the choice is recorded (`forecast_rossmann_test.md`, `forecast_weekly_test.md`; the Olist experiments append a test section) |
