# 0012: Step 5 classical model choices

- Status: Accepted. Choices written on 2026-10-24 from validation only, **before** any test split was scored. Test results are appended below, not used to change the choices.
- Module / task: M3, M5, M6, M7; Tasks 5, 6

Full tables: `docs/results/<experiment>.md`. Numbers here are validation unless marked test.

## 1. Rossmann daily forecaster (classical benchmark for decision 0005)

| Model | RMSPE (validation, 2015-05-09..06-19) |
| --- | --- |
| Store mean | 0.2995 |
| Same weekday, last 4 weeks | 0.2280 |
| Same day last year (rows with history) | 0.1814 |
| Ridge (alpha 0.1) | 0.1510 |
| LightGBM (127 leaves, 600 trees), 3 seeds | 0.1335 ± 0.0017 |
| **Random forest (300k rows), 3 seeds** | **0.1191 ± 0.0003** |

**Choice: random forest** is the classical benchmark the deep models must beat in Step 8 (decision 0005 named LightGBM as the expected benchmark; the rule is "best on validation", and the forest won by more than 8 standard deviations).

Why LightGBM lost, as far as we can tell: more trees made it worse (600 -> 1500 trees: 0.134 -> 0.143), so it fits patterns of the training years that did not hold in spring 2015. Averaging many deep, decorrelated trees was more robust here. LightGBM was tuned on a 4-point grid only; a wider search might close the gap and is noted as future work, not done now (no more tuning on this validation set after seeing the result).

Feature selection did **not** help: keeping the 16 features with permutation importance > 0.001 raised RMSPE from 0.134 to 0.163. The calendar features (day, weekofyear, dayofyear, month, store_month_dev) are correlated, so permuting one alone looks harmless while dropping all of them removes the seasonal signal. Lesson recorded: permutation importance underrates correlated groups; all features are kept.

Top drivers (permutation importance): store x weekday average, promotion, recent level (28-day mean 6 weeks back), store mean, weekday, store closed the day before.

Interval: LightGBM quantile P10-P90 coverage 0.767 (target 0.80): slightly too narrow.

## 2. Weekly forecaster (what a founder's upload gets)

12 weeks ahead, MASE (seasonal, m = 52), validation origin 24 weeks before each series' end. 935 Rossmann stores, 1,604 UCI products.

| Method | Rossmann stores | UCI products | Pooled |
| --- | --- | --- | --- |
| Naive (last week) | 1.021 | 0.790 | 0.875 |
| Moving average (4 weeks) | 0.897 | 0.666 | 0.751 |
| Seasonal naive (52 weeks) | 0.492 | 1.033 | 0.834 |
| Seasonal naive, level-adjusted | **0.418** | 1.144 | 0.877 |
| ETS (damped trend, log) | 0.906 | 0.637 | 0.736 |
| LightGBM global (3 seeds) | 1.149 ± 0.035 | **0.620 ± 0.002** | 0.815 ± 0.014 |
| **Auto: per series, the method that did best on its own last 12 weeks** | 0.467 | 0.703 | **0.616** |

**Choice: Auto** for the product's weekly forecast. No single method wins both kinds of business: seasonal methods win for stores with a stable yearly pattern, smoothing methods for sparse products. Auto picks per series (Rossmann: seasonal methods 94% of the time; UCI: ETS, naive or moving average 85%) and is best pooled. It also works on a single founder's upload, which a global model trained on someone else's data cannot promise.

Two bugs found and fixed during this experiment (first run, not used for the choice): the global LightGBM was trained on raw scaled targets, so sparse products with 500x jumps dominated the squared loss and every forecast (MASE 3.2 and 9.3); it now trains on log1p of the scaled target. The level-adjusted seasonal baseline divided by near-zero last-year sales (MASE 5.1 on UCI); its ratio is now capped at 0.25 to 4.

Intervals: Auto's band from its own backtest errors covers 0.946 (Rossmann, too wide) and 0.587 (UCI, too narrow) against 0.80; twelve backtest errors are too few to estimate the 10th and 90th percentiles. LightGBM quantile bands are worse (0.339, 0.719). Calibration is open work.

Global LightGBM still fails on Rossmann stores weekly (1.149): its features have no promotion calendar, and Rossmann's weekly sales follow the alternating promo weeks. It is a candidate for Step 8 comparisons, not for serving.

**Tier check (decision 0004).** With 8 to 52 weeks of history, the best available methods (moving average 0.897, ETS 0.93) do not improve as history grows: short-history forecasts are as good as smoothing allows, which supports offering a basic forecast from 8 weeks. Seasonal methods need 52 weeks (0.492) and the level-adjusted one 64 (0.418), roughly halving error: this supports reserving seasonal forecasts for the Full tier (52+ weeks).

## 3. Competitor-move classifier (decision 0006)

| Model | Macro-F1 | Brier |
| --- | --- | --- |
| Majority (stagnate) | 0.193 | n/a |
| Same move as last month | 0.212 | n/a |
| Logistic regression | 0.342 | 0.662 |
| Linear SVM (calibrated) | 0.197 | 0.510 |
| Decision tree (depth 6) | 0.254 | 0.774 |
| LightGBM | 0.362 ± 0.001 | 0.559 |
| Soft vote (logistic + forest) | 0.379 ± 0.003 | 0.594 |
| **Random forest** | **0.379 ± 0.003** | **0.566** |

**Choice: random forest** (tied with the soft vote on macro-F1, better Brier, simpler).

Findings that change the product:
- **Price cuts cannot be predicted.** With the same-product definition they are only 2% of seller-months (the draft's 25% was product-mix noise). F1 for price_cut is 0.00. The rule engine must treat a rival's price-cut risk as the base rate (about 2%), not as a model output.
- Exit (F1 0.45) and expansion (0.36) are predicted better than chance but are far from certain: probabilities, not verdicts.
- Accuracy rises with seller size (small 0.50, large 0.68) and history length (0.53 for 3 months or less, 0.67 for over a year): rivals with short histories are the hardest to read.
- The calibrated linear SVM collapses to the majority class after calibration (argmax ignores the class weights); kept in the table as a negative result.
- Feature selection (top half by mutual information) changed macro-F1 by +0.001: no meaningful gain; all features kept.

## 4. Market share (next month, categories with 5+ sellers)

| Model | MAE (pct points) | Small | Medium | Large sellers |
| --- | --- | --- | --- | --- |
| Persistence | 2.164 | 0.761 | 1.513 | 4.219 |
| Ridge | 2.170 | 1.174 | 1.675 | 3.661 |
| Random forest | 1.851 ± 0.001 | 0.802 | 1.421 | 3.334 |
| **LightGBM (L1 loss)** | **1.678 ± 0.006** | **0.639** | **1.250** | **3.140** |

**Choice: LightGBM with L1 loss** (22% lower error than persistence, and better in every size band).

## 5. Market pulse (trend regime, 12 weeks ahead)

| Model | Macro-F1 |
| --- | --- |
| Recent slope | 0.500 |
| Same weeks last year | 0.514 |
| Logistic regression | 0.426 |
| Random forest | 0.541 ± 0.013 |
| **LightGBM** | **0.551 ± 0.003** |

**Choice: LightGBM.** Gain over "same weeks last year" is small (+0.04). Limitation: one validation origin per series, so the score describes one season (validation labels are 60% falling, training 18%).

## 6. Peer cohorts (clustering)

**Choice: K-means, k = 3** (silhouette 0.187, stability ARI 0.99; Ward was unstable, ARI about 0.3-0.4). Segments: growing stores with a strong December; promotion-driven weekday stores; large steady stores. Structure is weak (silhouette below 0.25) and unrelated to StoreType (ARI 0.08): behaviour segments are not the same as format segments. Extreme stores (opened inside the window) are capped at the 1st/99th percentile so they do not form their own cluster.

## Test results (scored once, after the choices above)

### Rossmann daily forecaster (test window 2015-06-20..07-31, refit on everything before)

| Model | RMSPE |
| --- | --- |
| Same day last year (rows with history; the logged bar) | 0.1604 |
| Same day last year, else same weekday 4 weeks (all rows) | 0.1741 |
| Ridge | 0.1725 |
| **Random forest (chosen)** | **0.1300 ± 0.0001** |
| LightGBM | 0.1282 ± 0.0010 |

The forest beats the logged bar (0.160) by 19% and the all-rows baseline by 25%. On test, LightGBM is marginally ahead (0.128 vs 0.130, within 2 sd of each other's spread); the two are effectively tied, and the validation choice stands. Step 8's deep models must beat **0.130 on test / 0.119 on validation**. Intervals are too narrow: LightGBM quantile P10-P90 covered 0.654 on test (0.767 on validation) against a 0.80 target; interval calibration is a Step 8/9 task.

### Weekly forecaster (test origin 12 weeks before each series' end)

| Method | Rossmann stores | UCI products | Pooled |
| --- | --- | --- | --- |
| Moving average (4 weeks) | 1.036 | **0.920** | 0.963 |
| ETS | 1.036 | 0.923 | 0.964 |
| Seasonal naive, level-adjusted | **0.440** | 1.868 | 1.342 |
| LightGBM global | 0.573 ± 0.009 | 1.019 ± 0.016 | 0.855 ± 0.007 |
| **Auto (chosen)** | 0.448 | 1.020 | **0.810** |

Auto stays best pooled (16% below the moving average) and close to the best on Rossmann. On UCI products it is **worse than a plain moving average** (1.020 vs 0.920): the UCI test window is the September-November wholesale peak, and the method that won each product's previous 12 weeks (summer) often did not suit the peak. Products are the closer match to a small online seller, so this is the weakness to fix: Step 8 should test a seasonal-aware selection window (backtest on the same weeks last year) before claiming Auto for sparse products. Coverage of Auto's band: 0.695 Rossmann, 0.536 UCI (target 0.80): too narrow on test, so the UI must not call the band "80%" until it is calibrated.

### Competitor moves (test: feature months 2018-03..05, refit on train + validation)

| Model | Macro-F1 | Brier |
| --- | --- | --- |
| Same move as last month (best baseline) | 0.228 | n/a |
| **Random forest (chosen)** | **0.366 ± 0.004** | **0.563** |
| Soft vote | 0.371 ± 0.001 | 0.614 |
| LightGBM | 0.352 ± 0.005 | 0.595 |

The choice holds: +0.14 macro-F1 over the best baseline, best calibration. The soft vote is marginally higher on test but was not chosen and is less calibrated; no change.

### Market share (test)

| Model | MAE (pct points) |
| --- | --- |
| Persistence | 1.986 |
| **LightGBM L1 (chosen)** | **1.554 ± 0.007** |

The choice holds: 22% lower error than persistence, as on validation.

### Market pulse (test) — the chosen model fails

| Model | Macro-F1 |
| --- | --- |
| Recent slope | 0.481 |
| **Same weeks last year (rule)** | **0.618** |
| Random forest | 0.627 ± 0.003 |
| LightGBM (chosen on validation) | 0.543 ± 0.011 |

LightGBM, chosen on validation, is **worse than the "same weeks last year" rule on test**. The random forest edges the rule on test but lost on validation, so picking it now would be choosing on the test set. Verdict under the project rule "a model ships only if it beats its baseline": **market pulse ships as the deterministic "same weeks last year" rule** for uploads with 64+ weeks (52 + 12), and "recent slope" for shorter ones. The learned classifiers stay documented as experiments. Cause: one evaluation origin per series makes each split a single season, and the label mix swings (falling: 18% train, 60% validation, 21% test); a classifier trained on one season mix does not transfer. Rolling-origin evaluation would be needed before trying again.
