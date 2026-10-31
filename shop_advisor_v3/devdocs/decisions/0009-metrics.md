# 0009: Primary metrics

- Status: Accepted
- Date: 2026-10-24
- Module / task: Tasks 6, 7, 10

| Problem | Metric | Why this | Why not the obvious alternative |
| --- | --- | --- | --- |
| Rossmann forecast | RMSPE on open days with sales > 0 | The competition's official metric, so results are comparable with published work; percentage error is fair across big and small stores | MAPE weights all errors linearly; RMSE is dominated by large stores |
| Any forecast, cross-dataset | MASE (scaled by seasonal-naive error on training data) | Scale-free and defined when values hit zero; MASE below 1 means better than seasonal naive | MAPE blows up near zero sales, which is common for small sellers |
| Forecast band | P10 to P90 coverage, pinball loss | The founder sees a range; it must contain the truth about 80% of the time | Point error alone ignores whether the range is honest |
| Competitor move | Macro-F1 | Classes are imbalanced; macro-F1 weights rare moves (exit, expansion) equally | Accuracy rewards always predicting stagnate |
| Probabilities fed to rules | Brier score | Certainty factors must mean something | — |
| Market share | MAE in percentage points | Directly readable ("off by 2 points") | RMSE is harder to explain |
