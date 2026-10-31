# Demo + code walkthrough (what to say, what to show, card by card)

Use this with the website open in one window and VS Code in the other. For each card on the screen:
**say** the line, **show** the code (file:line), and if asked, give the **numbers** and the **difficulty we hit**.
Every number here comes from `docs/results/` and `devdocs/evaluation_log.md`.

Example used below: `demo/files/02_stockout_before_festival.csv` or your accessories-shop upload
(cash 200,000; the screen showed runway 1.7 months, peak 1 week away, Phone cases 52% of sales).

---

## 0. One-minute overview (before uploading)

**Say:** "Startup IQ is a business checkup. It has three kinds of intelligence, exactly the syllabus split:
**deep learning** learns patterns in sales sequences, **machine learning** predicts (forecast choice, competitor moves, market share),
and **AI reasoning** decides (a knowledge model, 47 IF-THEN rules, and A* search for a plan)."

**Show:** `devdocs/architecture.md` (the flow diagram) and `src/startupiq/pipeline.py:43` `analyze_intake()`: one function calls every module in order:
intake -> vitals -> forecast -> facts -> rules -> safety gate -> planner -> cards.

---

## 1. Card "Your data" (Task 4: load, inspect, clean)

**Screen:** "94 weeks read · Columns used: date = Bill Date, revenue = Amount (₹), product = Item · The week of 2024-10-27 is 2.2x a typical week. One-off bulk order?"

**Say:** "The founder uploaded a file with Indian dates and rupee amounts with commas. Nobody told the system which column is which: it scores every column by its **name** (Hindi or English synonyms) and by its **content** (do 80% of values parse as dates? are they numbers?). Then it cleans: duplicates, returns, partial weeks. It **asks** instead of guessing silently: here it found one week 2.2x normal and asks if it was a one-off."

**Show:**
- `src/startupiq/intake/columns.py:111` `assign_roles()`: name score x content score -> confidence; below 0.75 it asks.
- `src/startupiq/intake/parsing.py` `to_number()` (₹, lakh commas, `Rs. 450/-`) and `to_dates()` (ISO, day-first, US, Excel serial; ambiguous -> asks).
- `src/startupiq/intake/series.py:77` `build()`: duplicates, returns netted, partial weeks dropped, gaps and outliers turned into questions, data tier.

**Numbers:** 25 of 25 deliberately broken files handled correctly (`tests/fixtures/intake/`); real Rossmann, UCI and Olist exports mapped correctly.

**Difficulty -> fix (true stories):**
1. Identical rows in a shop register were deleted as duplicates, but two customers can buy the same item the same day. Fix: judge duplicates on the **whole row**; keep them in shop logs without timestamps.
2. On the real Olist export it picked the *shipping deadline* as the sale date and *freight* as revenue. Fix: prefer "order/purchase/invoice" dates, penalise "ship/deliver/freight".
3. Monthly sheets with one column per month were read as sparse weekly data. Fix: detect "periods across" and reshape.

**Datasets for Task 4** (if asked): Rossmann (1.02M store-days), UCI Online Retail II (1.07M lines), Olist (112k items, 3,095 sellers), FRED + Wikipedia. Cleaning log with before/after counts: `docs/results/cleaning_log.md`; 15 EDA figures: `docs/results/eda_summary.md`.

---

## 2. Card "Your next 3 months" (Tasks 6 + 7 + 8: ML and DL forecasting)

**Screen:** "About 739,673 expected over 12 weeks (likely range ...) · Method: auto: Seasonal naive (52 weeks)"

**Say:** "This is the forecast. We tried many models and let the **data decide** per business. 'Auto' backtests five classical methods on *this shop's own last 12 weeks* and keeps the winner. Here it chose 'same weeks last year', because this shop is very seasonal. We also trained **deep learning models** on 2,539 real series (Rossmann stores and UCI products) on a GPU; they are added to Auto as one more candidate: the **hybrid**."

**Show (ML):** `src/startupiq/models/forecast_methods.py:56` `auto()` (backtest, pick the winner, band from backtest errors).

**Show (DL, Task 7):**
- `src/startupiq/dl/models.py:69` `LSTMForecaster` (LSTM encoder + attention over 52 weeks), `:113` `TCNForecaster` (1D-CNN: causal dilated convolutions, dilations 1..32, sees 127 weeks), `:56` `MLPForecaster` (the basic neural model).
- `src/startupiq/dl/models.py:139` `pinball()`: quantile loss, so the model predicts a low, middle and high value (P10/P50/P90); `Decoder` (`:27`) makes them never cross.
- `src/startupiq/dl/train.py:81` `fit()`: AdamW, gradient clipping, learning-rate halving, early stopping on 10% of series held out.
- `src/startupiq/dl/windows.py`: each training example = 52 past weeks -> next 12 weeks, scaled by the recent level so a drugstore and a tiny product can share one model.

**Numbers (MASE, lower is better; 1.0 = as good as "same week last year" in-sample):**

| | Validation, all series | Validation, sparse products | Test, all | Test, sparse products |
| --- | --- | --- | --- | --- |
| Auto (classical) | **0.616** | 0.703 | **0.810** | 1.020 |
| TCN (1D-CNN) | 0.686 | **0.600** | 0.861 | **0.908** |
| LSTM + attention | 0.716 | 0.607 | 0.914 | 0.948 |
| MLP (basic) | 0.720 | 0.617 | 0.927 | 0.927 |
| **Hybrid (Auto + TCN)** | **0.603** | 0.682 | | |

Daily Rossmann (RMSPE): TCN 0.131 beats LightGBM 0.134, but the random forest is best at 0.119.

**Say the honest conclusion:** "Deep learning alone did not beat the classical choice overall, because stores have a strong yearly pattern that 'same weeks last year' captures perfectly. But on small, irregular products, the kind of business our users run, every deep model won, on validation and on test. So we did not throw it away: the hybrid uses the TCN only where it backtests better, and that beat Auto (0.603 vs 0.616)."

**Difficulty -> fix (say these, examiners love them):**
1. **CPU too slow:** one LSTM took 2.5 hours on the CPU-only PyTorch. We installed the CUDA build and trained on the RTX 2050: 25x faster (11.6 s per epoch).
2. **The weekday bug:** the first daily model scored 0.549 (worse than an average). Every training window started 14 days apart, so all started on a **Tuesday**; the model learned the weekday from position and was 30-60% off at a Saturday start. Found by scoring day by day (05-05 fine, 05-09 broken). Fix: start every 10 days so all weekdays appear. Result 0.145. A test now guards it (`tests/test_dl.py`).
3. **One bad series ruined everyone:** sparse products with sudden 500x jumps dominated the squared loss (MASE 3-9). Fix: train on log(1 + value / level).
4. **GPU and page-file out of memory** when three jobs ran at once: run one at a time, smaller evaluation batches, free GPU memory after each model.
5. **Attention helps:** LSTM 0.695 with attention vs 0.776 without. **Dropout 0.3** made the ranges honest (coverage 0.74 -> 0.83).

**Task 8 interpretability (show `docs/results/dl_interpret.md`, figures 18-21):**
- Saliency (gradient x input): the TCN puts 43% of its attention on the last 4 weeks for stores, 16% on "same week last year".
- **Surprise:** LSTM attention weights do **not** match what really changes the forecast (occlusion: Spearman -0.11), while gradient saliency does (0.63). So "attention = explanation" is not true here; that is why the product explains forecasts with backtests and saliency.
- TCN first-layer filters fire on "a sudden spike in the latest week" (figure 20).

---

## 3. Card "Your money" (Task 1 knowledge model + arithmetic)

**Screen:** "Runway: 1.7 months · cash 200,000 / monthly burn 115,283 (fixed costs minus gross profit)"

**Say:** "No AI here on purpose: runway is cash divided by burn. Using a model for an exact quantity would only add error. The same formula lives in our **knowledge model** as an 'if-needed' slot of the Business frame."

**Show:** `src/startupiq/vitals.py:29` and `src/startupiq/knowledge/ontology.py` `_runway()` attached as `if_needed` on the Business frame; `src/startupiq/knowledge/frames.py:55` `get()` (own value -> if-needed -> parent -> default).

---

## 4. Card "What will go wrong" (Task 2: rule engine, forward chaining)

**Screen:** warnings (cash 1.7 months; busiest week 1 week away; Phone cases 52%), diagnoses ("riding the season", "growing but cash-starved"), 90-day pre-mortem.

**Say:** "Every number from the models becomes a **fact** with a certainty. 47 IF-THEN rules then reason over them by **forward chaining**. Example from the trace: R10 'runway < 3 months' -> cash_risk = high; that new fact triggers R13 (forbid a new product or channel), R14 (cut costs) and R70 (growing but cash-starved). Rules fire by priority, then by how specific they are."

**Show:**
- `configs/rules.yaml:75` (R10), `:113` (R13), `:461` (R70): rules are plain data, each with a firing and a non-firing test example.
- `src/startupiq/rules/engine.py:156` `run()`: agenda, conflict resolution, refraction, certainty = rule cf x weakest condition (MYCIN).
- Website: open "Full reasoning trace": the exact firing order.

**How the rules were made (say it exactly like this):** "These are **expert rules we wrote**, which is what Task 2 asks for. Their thresholds come from the data: for example, rising or falling is ±10% because on Rossmann only 20% of 4-week changes exceed 10%; 'share slipping' is a 2-point drop because that is the share model's error and 69% of such alerts were right." The decision tree is a different thing: in the competitor model (Task 6) we trained a decision tree and **printed its rules** for readability (`docs/results/competitor.md`, 'Readable rules'); we did not copy them into the rule base.

**Numbers:** 47/47 rules fire on their example and stay silent on their counter-example; 47/47 expectations met in 17 founder scenarios (written by us, so a consistency check); rules fire on 55% of 80 real uploads.

**Difficulty -> fix:** on real Olist sellers nearly everyone was diagnosed "losing share". Checked the model: not biased, alerts right 69% of the time, but the threshold (0.5 points) was below the model's error. Raised to 2 points: alerts dropped from 34% to 14% at the same accuracy. Also: "keep things as they are" was shown next to a visible problem when finances were missing, so we added R84 ("give us your finances first").

---

## 5. Card "Your market" (Task 6 result that a model lost)

**Screen:** "Next weeks look rising (same weeks last year: +17%)"

**Say:** "We trained classifiers (LightGBM, random forest) to predict rising / flat / falling. On test, the chosen model scored 0.543 while the simple rule 'what happened in the same weeks last year' scored 0.618. Our rule was fixed before testing: a model ships only if it beats its baseline. So the **rule ships**, and we report the model as a failed experiment. That is honest ML."

**Show:** `src/startupiq/rules/facts.py:63` `market_pulse()`; `devdocs/decisions/0012-step5-model-choices.md` section "Market pulse (test): the chosen model fails".

---

## 6. Card "Your competitors" (Tasks 5 + 6: classification)

**Screen:** "2 rivals; estimates, not observations", probabilities of exit / expansion / price cut.

**Say:** "A founder cannot see a rival's sales, so this is labelled **inferred**. We trained it on Olist, a marketplace where every seller is visible: for each seller-month we built features (orders, share, price position, reviews...) and labels from the next 3 months: exit, price cut, expansion, stagnate. A random forest predicts the move."

**Show:**
- `src/startupiq/features/olist.py:39` `build()`: features use only months up to t; labels look at t+1..t+3 (no leakage; `devdocs/leakage_checklist.md`).
- `src/startupiq/models/competitor.py:51` `make_models()`: logistic regression, SVM, decision tree, random forest, LightGBM, voting.

**Numbers (macro-F1, test):** random forest 0.366 vs baseline "same move as last month" 0.228 (accuracy higher for big sellers 0.68 than small 0.50).

**Difficulty -> fix:** at first 25% of seller-months looked like "price cuts". Measured on the *same product*, real price cuts are only 2%: the rest was a change in which products sold. With 2%, price cuts cannot be predicted (F1 0), so the app says so honestly and uses the base rate.

---

## 7. Card "What to do next" (Tasks 1 + 2 + 3: constraints, backward chaining, A* search)

**Screen:** "Cut a fixed cost (rule R14)" · "Why not other moves?" · "No combination of up to 4 allowed moves reaches the target within the cash available."

**Say (recommendation):** "R14 recommends cutting a cost. Before showing any move, a **safety gate** checks it against the knowledge model's constraints: does cash cover it, does margin stay above the floor, is there stock, is it legal. Agreeing prices with a rival is in the ontology only so it can always be refused."

**Say (why not):** "Click 'why not other moves'. That is **backward chaining**: it starts from the goal 'recommend expansion' and walks back to the conditions that failed, e.g. 'healthy growth does not follow because cash risk is high'."

**Say (plan):** "The planner is **state-space search**: a state is a plan of up to 4 moves over 12 weeks; a simulator plays it on the forecast; the goal is +3% of sales in profit with cash never below zero; cost is the cash spent plus margin given away. A* finds the cheapest plan. Here, with 1.7 months of cash, **no plan exists**: the search explored 361 plans and proved it. Telling a founder 'there is no safe way to grow right now, cut costs first' is the correct answer."

**Show:**
- `src/startupiq/knowledge/kb.py:78` `check()` and `src/startupiq/knowledge/constraints.py` (7 constraints).
- `src/startupiq/rules/engine.py:196` `prove()` and `:232` `why_not()` (backward chaining).
- `src/startupiq/search/problem.py:102` `simulate()`, `:161` `successors()` (only moves the constraints allow), `:178` `heuristic()` (fractional knapsack); `src/startupiq/search/algorithms.py:42` BFS, `:62` DFS, `:105` UCS, `:109` A*, `:113` greedy.

**Numbers (12 planning problems, `docs/results/planner.md`):** A* optimal 10/10 (always equal to uniform-cost search) with 5-50x fewer nodes on medium targets; BFS optimal 3 (fewest moves, not cheapest); DFS 2; greedy 2 (fast, wrong cost).

**Difficulty -> fix:** (1) the simulator first never re-ordered stock, so "do nothing" sold out and every target looked impossible: added routine reorders. (2) The heuristic measured the gap at optimistic P90 demand while the goal used P50, so it was always 0 and A* was no faster than uniform-cost: fixed to use the same simulator.

---

## 8. Close: validation and ethics (Task 10)

**Say:** "We tested on 300 Olist sellers the system never saw. It beats 'repeat last week' (0.982 vs 1.036) but not a plain moving average (0.936), and its ranges contain the truth 56% of the time, not 80%, so we call them 'likely range'. It is fairer to small sellers (0.917) than large ones (1.105). A full analysis takes 0.26 seconds and the same upload always gives the same answer. Files are processed in memory and never stored; illegal advice is refused; estimates are labelled."

**Show:** `docs/results/validation.md`, `docs/ETHICS.md`, `docs/REPORT.md`.

---

## Likely viva questions (short answers)

| Question | Answer |
| --- | --- |
| Why not use only the LSTM? | It lost to the classical choice overall (0.716 vs 0.616); it only wins on sparse products, so it is one candidate in the hybrid. |
| Why LSTM *and* CNN? | Syllabus asks for CNN/RNN/LSTM; they learn differently (memory vs local patterns). The TCN was better and 4x faster to explain via filters. |
| How do you know there is no data leakage? | Every feature uses data before the forecast start; a test multiplies all future sales by 10 and checks no feature changes (`tests/test_features.py`). Test sets were scored once, after choices were written down. |
| Where did the rules come from? | Written as expert rules (Task 2) with data-backed thresholds; the decision tree's printed rules are a separate Task 6 output. |
| Forward vs backward chaining here? | Forward: from facts to warnings and recommendations. Backward: from "recommend X" back to the missing or false conditions ("why not X"). |
| Is A* optimal? | Only if the heuristic never overestimates; we did not assume it, we checked A* against uniform-cost search: equal cost in 12 of 12 problems. |
| What would you improve? | Calibrate ranges (56% -> 80%), pilot with real Indian shops, local category data. |
