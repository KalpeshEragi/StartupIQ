# Evaluation protocol

Fixed before any model is trained. Changing it needs a decision record.

## Splits

| Data | Split | Why |
| --- | --- | --- |
| Rossmann | Train up to 2015-06-19, test 2015-06-20 to 2015-07-31 (last 42 days); validation is the 42 days before the test | Mirrors the real use: forecast the future from the past. Random splits leak future information into training. |
| UCI Online Retail II | Last 12 weeks held out | Same reason; shorter series |
| Olist | Feature month t: train 2017-03 to 2017-11, validation 2017-12 to 2018-02, test 2018-03 to 2018-05. Labels look at t+1..t+3, so each split's labels end before the next split starts (test labels reach 2018-08, the end of the data) | Temporal only, not grouped by seller: knowing a rival's past is the real use case. Changed on 2026-10-24 from the draft (which put test at 2018-06..08) because 3-month exit labels need 3 months after t |
| Weekly panel (Rossmann stores, UCI products) | Per series: test origin 12 weeks before the end, validation origin 24 weeks before; training origins every 2 weeks with targets ending before the evaluation origin | One origin per series means validation and test each cover one season (late winter for Rossmann, summer for UCI). Reported as a limitation |
| Cross-dataset | Leave one dataset out: train on two, test on the third | Measures how the forecaster behaves on a business it has never seen (the founder case) |

Model selection currently uses one validation origin per dataset (Step 5). Rolling-origin backtests over several cut-off dates were planned but are not implemented yet; until they are, a validation result describes one season. The test window is scored **once**, with `python -m startupiq.models.run <name> --final`, after the choice is written in the decision record.

## Baselines every model must beat

| Problem | Baselines |
| --- | --- |
| Forecast | Store/series mean; same weekday of the last 4 weeks; same period last year (seasonal naive) |
| Competitor move | Majority class; "same as last month" |
| Market share | Persistence (next share = current share) |
| Trend regime | Sign of the 8-week slope |

Current numbers for these are in `evaluation_log.md`.

## Metrics

Decision [0009](decisions/0009-metrics.md) explains each choice.

| Problem | Primary | Also reported |
| --- | --- | --- |
| Forecast | RMSPE on open days with sales > 0 (Rossmann), MASE (all datasets) | MAE, sMAPE, P10 to P90 coverage, pinball loss |
| Competitor move | Macro-F1 | Accuracy, per-class precision and recall, confusion matrix |
| Market share | MAE in percentage points | Error by seller size band |
| Clustering | Silhouette, Davies-Bouldin | Stability across seeds (ARI) |
| Rules | Agreement with hand-labelled scenarios | Rule coverage (share of rules that ever fire) |
| Search | Plan value, nodes expanded, time | Gap to exhaustive search on short horizons |

## Repeats and noise

Every learned model runs with 3 seeds; report mean and standard deviation. Two models whose means differ by less than one standard deviation are treated as tied, and the simpler one wins.

## Fitting hygiene

Scalers, encoders and feature selection are fitted on training data only, inside a pipeline. Lag and rolling features are computed only from data before the forecast origin. Leakage checklist lives in Step 4's decision record.
