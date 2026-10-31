# Startup IQ: "Startup Doctor" full implementation plan (CIL Lab CELBC710, Tasks 1 to 10)

Version 3. Supersedes `IMPLEMENTATION_PLAN.md` (YC/Form D direction) and earlier drafts of this file.

## 1. What we are building

A founder uploads a sales sheet and gets a **weekly briefing**: forecast, market direction, stock and cash warnings, likely competitor moves, estimated market share, and a recommended plan, each with the reason shown. Think of it as a checkup for a small business: vitals, prognosis, diagnosis, treatment plan, and a "how this could fail" list.

**Not a success predictor, not a guarantee.** Forecasts come as ranges. Competitor and market-share outputs are inferred and labelled as such. The system refuses to answer when data is too thin.

## 2. Modules and technique used

Rule: use the simplest technique that is correct. Arithmetic stays arithmetic.

| ID | Module | Technique | Task |
| --- | --- | --- | --- |
| M0 | Intake: column detection, validation, cleaning, quality tier | Deterministic rules | 4, 5 |
| M1 | Vitals: runway, margin, growth, stock cover, product dependence | Deterministic formulas | 1, 2 |
| M2 | Anomaly radar: odd days, stockouts, spikes | ML (Isolation Forest); DL autoencoder is a stretch | 6 |
| M3 | Prognosis: sales forecast with P10/P50/P90 | **DL** (LSTM, 1D-CNN) vs ML (LightGBM) vs baselines | 6, 7, 8 |
| M4 | Voice of customer: review text to sentiment and themes | **DL** text CNN (stretch) | 7, 8 |
| M5 | Market pulse: trend regime from own series plus public demand | ML plus decomposition | 6 |
| M6 | Peer cohort: "businesses like yours" | ML clustering | 6 |
| M7 | Rival radar: competitor move probabilities, inferred market share | ML classifier and regressor | 5, 6 |
| M8 | Diagnosis and pre-mortem | **Rule-based expert system** (forward and backward chaining, certainty factors) | 1, 2 |
| M9 | Treatment planner with what-if simulator | **Search** (BFS, DFS, A*, greedy) over the M3/M7 models | 3 |
| M10 | Safety gate: abstain on thin data, block illegal advice, subgroup audit | Rules plus audit | 10 |
| M11 | Doctor's note: plain-language summary with evidence | Templates over rule trace and saliency | 8, 10 |

```
upload -> M0 -> M1 vitals, M2 anomalies
            -> M3 forecast, M4 reviews, M5 trend, M6 peers, M7 rivals   (all become FACTS with confidence)
            -> M8 diagnosis (rules) -> M9 plan (search, simulated) -> M10 safety gate -> M11 note
```

ML and DL outputs enter the rule engine as facts with a certainty factor. This is Task 9's "DL for representation, ML for prediction, AI for decision making".

## 3. Intake and data-quality tiers (M0)

Required: a date and an amount. Optional: product, price, cost, stock, channel, customer. Each optional column unlocks one more card.

Reading: detect columns by name and by content, handle title rows, wide month-per-column sheets, DD/MM vs MM/DD, currency symbols; ask the user to confirm low-confidence guesses; offer a 3-column template; later add adapters for Shopify, Amazon/Flipkart and UPI exports.

| History given | Gets | Switched off |
| --- | --- | --- |
| Under 8 weeks | Summary and simple trend | Forecast, seasonality |
| 2 to 6 months | Simple-method forecast, wide range, "low confidence" | Neural network, seasonal claims |
| 6 to 12 months | Better forecast, weekly pattern, stock and cash rules | Yearly seasonality |
| 12+ months with product and price | Full forecast, product view, what-if | Nothing |

Refusals: all zeros, no date column, fewer than about 20 rows, nonsense file. Each refusal says what is needed. Cleaning is logged and shown. A pack of about 15 deliberately broken files is the regression test.

## 4. Datasets (checked)

| Role | Dataset | Verified facts | Licence note |
| --- | --- | --- | --- |
| Forecast training and baselines | Rossmann (Hugging Face mirror `AiiN-aini/rossmann-store-sales`) | 1,017,209 rows, 1,115 stores, 2013-01 to 2015-07, no duplicates or missing values in `train.csv`; promo lift 1.39x; 6-week holdout RMSPE: store mean 0.331, same weekday last 4 weeks 0.264, same day last year 0.160 | Unofficial re-upload, no licence stated; original under Kaggle rules. Lab use only. |
| Forecast generalisation, intake tests | UCI Online Retail II | 1,067,371 rows, 5,305 products, 2009-12 to 2011-12, 1,648 products with 52+ active weeks, weekly revenue 12-week holdout MAPE: naive 0.329, seasonal naive 0.218 | CC BY 4.0 per UCI listing |
| Competitors and market share | Olist (Hugging Face mirror `miminmoons/olist-...`) | 99,441 orders, 3,095 sellers, 2016-09 to 2018-10; from 2017-01 to 2018-08, 751 category-months with 5+ sellers; 1,361 seller-category pairs active 6+ months; share persistence baseline MAE 2.2 percentage points | Mirror says MIT but original is CC BY-NC-SA 4.0. Non-commercial, attribute, share-alike. |
| Market trend | FRED retail series (RSXFS, RSCCAS, MRTSSM4541USN), Wikipedia page views | Monthly series; Wikipedia articles can be short (Wireless_earbuds has 25 months) | Open, cite |
| Customer reviews (stretch) | Olist reviews | 99,224 reviews, 41% with text, Portuguese | As Olist |
| Real pilot | 3 to 5 small businesses, with consent | Collected by the team | Consent form; data never leaves the team |

All downloads are in `data/raw/` and `MANIFEST.md`. YC and Form D files stay but are not used.

**Known data issues to handle:**
- Rossmann has no public test labels, so we hold out our own last 6 weeks and cannot compare with the Kaggle leaderboard.
- Olist "price cut" from median price is contaminated by product mix; define it on the same product's price. "No orders next month" is mostly a sparse month, not an exit; define exit as no orders for 3 months and never again.
- Rossmann stores are German drugstores; founder data is mostly online small sellers. Use leave-one-dataset-out evaluation, and train the global model on a mix of Rossmann, UCI products and Olist seller series.

## 5. The steps

Thirteen steps: Step 0 plus twelve. Each ends with a gate.

| Step | Name | Tasks | Main work | Deliverable | Gate |
| --- | --- | --- | --- | --- | --- |
| 0 | Foundation | - | Dependencies in `pyproject.toml` (pandas, scikit-learn, lightgbm, statsmodels, torch, networkx, pyyaml, pydantic, fastapi, matplotlib, pytest), module folders, interface stubs, dummy `analyze()` | Stub pipeline returns a fake report; tests pass on a fresh clone | One command installs and tests |
| 1 | Data registry | 4 | Data cards for Rossmann, UCI, Olist, FRED, Wikipedia; licence table | `docs/data_cards/*.md` | All datasets in manifest with licence notes |
| 2 | Clean and explore | 4 | Scripted cleaning with before/after row counts; weekly and monthly tables; at least 10 plots | Cleaned tables in `data/processed/`, EDA notebook, quality log | Rebuilds from raw with one command |
| 3 | Intake module | 4, 5, 10 | Column detector, validator, tiering, confirmation step, template, broken-file pack | `src/startupiq/intake/`, 15 broken files, tests | Every broken file gets the right tier or refusal |
| 4 | Features and splits | 5 | Lags, rolling, calendar, promo, price, share features; competitor-move labels; feature selection; time-based splits grouped by series | Feature tables, leakage checklist | Checklist signed; transformers fitted on train only |
| 5 | Classical ML | 6 | Forecast baselines, ETS/ARIMA, ridge, random forest, LightGBM; move classifier; share regressor; clustering; trend regime; error analysis | One comparison table with metrics | Every model compared with a stated baseline; test set scored once |
| 6 | Knowledge model | 1 | Frames and semantic network for business, product, competitor, move, constraint; consistency checker; 8 labelled synthetic scenarios | `src/startupiq/knowledge/`, graph figure | Invalid scenarios rejected, valid pass |
| 7 | Rule engine | 2 | Own forward and backward chaining engine; 30 to 40 rules (vitals, diagnosis, pre-mortem, safety); trace output | `src/startupiq/rules/`, `configs/rules.yaml` | Each rule has a firing and a non-firing test |
| 8 | Deep learning | 7 | Global LSTM and 1D-CNN with quantile loss; regularisation table; random search; 3 seeds; decision rule vs LightGBM | `src/startupiq/dl/`, training curves, saved models | Decision rule applied and recorded |
| 9 | Interpretability | 8 | Gradient saliency, CNN feature maps, attention vs lag-ablation; comparison with baselines | Figures and a table (accuracy, calibration, interpretability, speed) | Table complete for every model |
| 10 | Planner | 3 | State space, simulator from forecaster and share model; BFS, DFS, A*, greedy; compare plan quality, nodes, time | `src/startupiq/search/`, comparison table | Checked against exhaustive search on a small horizon |
| 11 | Integration | 9 | `analyze()` wiring all modules; FastAPI; web UI with five cards, what-if sliders, doctor's note | Working app on demo data | End-to-end run returns all cards with evidence |
| 12 | Validation, ethics, deployment | 10 | Unseen and out-of-domain tests; messy-file tests; interval calibration; subgroup audit; pilot comparison; deploy; report and slides | Report, slides, deployed demo | Demo rehearsed on unseen files |

**Evaluation details for Step 12:**
- Unseen inputs: held-out last 6 weeks of Rossmann; leave-one-dataset-out (train without Olist or UCI, test on it); hand-made founder scenarios with expected outcomes; the broken-file pack.
- Metrics: RMSPE and MASE for forecasts; coverage of P10 to P90 (target about 80%); macro-F1 for competitor moves; share MAE in percentage points; rule agreement; latency.
- Fairness and ethics: error by seller size and category (small sellers likely worse), survivorship in Olist, one-country data, competitor surveillance limits (public data only), antitrust rule against price coordination, privacy (uploaded sheet not stored), overconfidence handled with ranges and abstention.

## 6. Dependencies and parallel tracks

Three tracks run side by side after Step 0.

```
Data track: 1 -> 2 -> 3 -> 4 -> 5 -> 8 -> 9
AI track:   6 -> 7 -> 10          (needs only a mocked forecast until Step 8 lands)
Product:    stub (0) -> 11 ... integration grows continuously
Final:      12
```

Step 10 needs a simulator, so start with the LightGBM forecaster from Step 5 and swap in the neural network after Step 8.

## 7. Time and people

| Team | Estimate |
| --- | --- |
| Four people in parallel | about 6 weeks |
| Two people | about 9 weeks |
| One person | about 12 to 14 weeks |

Approximate hours per step: 0 (4), 1 (4), 2 (12), 3 (14), 4 (10), 5 (16), 6 (8), 7 (14), 8 (18), 9 (10), 10 (14), 11 (18), 12 (20). Total about 160 hours. The syllabus allows 30 lab hours of contact time, so most work is outside lab sessions. I do not know your deadline; fit the schedule to it using the cut list below.

Suggested split for four: A takes steps 6, 7, 10. B takes steps 1 to 5. C takes steps 8, 9. D takes steps 0, 3, 11, 12 with everyone helping on 12.

Week plan: week 1 steps 0 to 2 and 6; week 2 steps 3, 4, 7; week 3 step 5 and finish 7; week 4 steps 8 and 10; week 5 steps 9 and 11; week 6 step 12.

## 8. Cut list if time runs short (drop from the bottom)

1. M4 review-text CNN and the autoencoder.
2. Wikipedia and FRED signals (trend uses only the user's series).
3. Cohort clustering UI (keep the clustering analysis for Task 6).
4. Shopify/UPI adapters (keep the generic reader and template).
5. Deployment beyond local run and a Dockerfile.

Never cut: M0 tiers, M3 with baselines, M8 rules, M9 search, M10 safety gate, the Task 10 report. These carry the syllabus tasks.

## 9. Repository layout

```
src/startupiq/
  ingest/ intake/ clean/ features/ models/ dl/
  knowledge/ rules/ search/ explain/ evaluate/
  pipeline.py  api/
configs/   rules.yaml, moves.yaml, model configs
data/raw/  rossmann_hf, olist_hf, uci_retail, fred, wiki (+ older yc, formd)
docs/      plan, data_cards, decisions, results
tests/     per module, plus broken-file pack
```

## 10. Risks

| Risk | Handling |
| --- | --- |
| Neural network does not beat LightGBM | Decision rule written beforehand; NN still counts as implemented, tuned and interpreted for Tasks 7 and 8 |
| Competitor and share outputs are inferred, not observed | Label as inferred, show ranges, report accuracy measured on Olist |
| Olist is sparse | Use categories with 5+ sellers, quarterly aggregation fallback |
| Transfer from Rossmann to a founder's data | Leave-one-dataset-out test, tier fallbacks, pilot data |
| Licences | Lab use only until original terms are checked; no redistribution of raw files |
| Scope | Cut list above; steps have gates so partial delivery is still coherent |
| Search with a learned simulator is not provably optimal | Verify against exhaustive search on a small horizon, state the assumption |

## 11. Definition of done

Each step is done when its code, tests, a short dated decision note, and its results table or figure are saved, and its gate has passed. The project is done when Step 12's demo runs on files nobody has seen before, including the broken-file pack, and the report documents limits and ethics.
