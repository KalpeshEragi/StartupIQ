# Evaluation log

Append-only. One row per measured result. Newest at the bottom. "Split" refers to `evaluation_protocol.md`.

| Date | Data | Split | Model | Metric | Value | Produced by | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-23 | UCI Online Retail II, weekly total revenue | last 12 weeks | Naive (last week) | MAPE | 0.329 | `scripts/exploration/profile_uci.py` | Exploration, before cleaning |
| 2026-10-23 | UCI Online Retail II, weekly total revenue | last 12 weeks | Mean of last 4 weeks | MAPE | 0.401 | same | Exploration |
| 2026-10-23 | UCI Online Retail II, weekly total revenue | last 12 weeks | Seasonal naive (52 weeks) | MAPE | 0.218 | same | Exploration; seasonality is strong |
| 2026-10-24 | Rossmann, 1,115 stores daily | last 42 days | Store mean (open days) | RMSPE | 0.331 | `scripts/exploration/rossmann_baselines.py` | 40,282 store-days scored |
| 2026-10-24 | Rossmann | last 42 days | Same weekday, last 4 weeks | RMSPE | 0.264 | same | |
| 2026-10-24 | Rossmann | last 42 days | Same day last year (364 days) | RMSPE | 0.160 | same | 35,077 store-days (stores without last-year data skipped). **Bar to beat.** |
| 2026-10-24 | Olist, seller share within category-month, categories with 5+ sellers | all consecutive months | Persistence | MAE (pct points) | 2.20 | `scripts/exploration/profile_hf.py` | Mean share 3.76; 12,834 pairs |
| 2026-10-24 | Broken-file pack, 25 synthetic files (`tests/fixtures/intake`) | all | Intake v1 | Files handled as expected (tier, roles, questions) | 25 / 25 | `pytest tests/test_intake_pack.py` | First run found 3 bugs (identical shop rows dropped, wide monthly read as weekly, products-as-columns not reshaped); fixed |
| 2026-10-24 | Real exports: Rossmann store 1, UCI raw lines, Olist top seller's items | all | Intake v1 | Correct date, amount and product columns | 3 / 3 | `pytest tests/test_intake_units.py` + check in decision 0011 | First run picked Olist shipping date and freight value, and treated multi-unit order rows as duplicates; fixed |
| 2026-10-24 | UCI raw lines through intake vs Step 2 dataset cleaner | 104 weeks | Intake v1 | Mean absolute % difference of weekly revenue | 0.033 | check in decision 0011 | Duplicates removed match exactly (34,335); gap is postage/fee lines intake cannot recognise in an unknown file |
| 2026-10-24 | Rossmann daily, cleaned | validation (2015-05-09..06-19) | Same day last year / Ridge / LightGBM / **Random forest** | RMSPE | 0.181 / 0.151 / 0.134 ± 0.002 / **0.119 ± 0.000** | `python -m startupiq.models.run forecast_rossmann` | Forest chosen (decision 0012). Feature selection hurt: 0.163 |
| 2026-10-24 | Rossmann daily, cleaned | **test** (2015-06-20..07-31), once | Same day last year / Ridge / **Random forest** / LightGBM | RMSPE | 0.160 / 0.173 / **0.130 ± 0.000** / 0.128 ± 0.001 | `... forecast_rossmann --final` | Bar for Step 8. Quantile P10-P90 coverage 0.654 (too narrow) |
| 2026-10-24 | Weekly panel: 935 Rossmann stores + 1,604 UCI products | validation (24 weeks before end) | MA4 / ETS / LightGBM global / **Auto** | MASE pooled | 0.751 / 0.736 / 0.815 ± 0.014 / **0.616** | `... forecast_weekly` | Auto chosen; now serves the product forecast card |
| 2026-10-24 | Weekly panel, Rossmann stores only | validation | Seasonal naive / level-adjusted / Auto | MASE | 0.492 / 0.418 / 0.467 | same | Seasonal methods halve error once 52+ weeks exist (tier check) |
| 2026-10-24 | Weekly panel | **test** (12 weeks before end), once | MA4 / LightGBM global / **Auto** | MASE pooled (UCI only) | 0.963 (0.920) / 0.855 (1.019) / **0.810 (1.020)** | `... forecast_weekly --final` | Auto best pooled but loses to MA4 on UCI's peak season; band coverage 0.70 / 0.54 |
| 2026-10-24 | Olist competitor moves (exit, price cut, expansion, stagnate) | validation (2017-12..2018-02) | Same move as last month / LightGBM / **Random forest** | Macro-F1 | 0.212 / 0.362 / **0.379 ± 0.003** | `... competitor` | Price cut (2% of rows) F1 = 0 |
| 2026-10-24 | Olist competitor moves | **test** (2018-03..05), once | Same move as last month / **Random forest** | Macro-F1 (Brier) | 0.228 / **0.366 ± 0.004 (0.563)** | `... competitor --final` | Choice holds |
| 2026-10-24 | Olist next-month share, categories with 5+ sellers | validation | Persistence / **LightGBM L1** | MAE pct points | 2.164 / **1.678 ± 0.006** | `... share` | |
| 2026-10-24 | Olist next-month share | **test**, once | Persistence / **LightGBM L1** | MAE pct points | 1.986 / **1.554 ± 0.007** | `... share --final` | 22% better, every size band |
| 2026-10-24 | Trend regime, weekly panel | validation | Same weeks last year / **LightGBM** | Macro-F1 | 0.514 / **0.551 ± 0.003** | `... regime` | |
| 2026-10-24 | Trend regime | **test**, once | **Same weeks last year** / LightGBM | Macro-F1 | **0.618** / 0.543 ± 0.011 | `... regime --final` | Chosen model loses to the rule on test: rule ships (decision 0012) |
| 2026-10-24 | Rossmann store segments | training window | **K-means k=3** / Ward k=3 | Silhouette (stability ARI) | **0.187 (0.99)** / 0.168 (0.37) | `... clustering` | Weak structure; unrelated to StoreType (ARI 0.08) |
| 2026-10-24 | Knowledge validation scenarios (8 synthetic) | all | Constraint checker | Scenarios with exactly the expected violations | 8 / 8 | `pytest tests/test_knowledge.py` | Two scenario-data mistakes found and corrected first |
| 2026-10-24 | Rule examples (47 rules) | all | Rule engine | Rules firing on their example / silent on their counter-example | 47 / 47, 46 / 46 | `pytest tests/test_rules.py` | R90 always fires, so it has no counter-example |
| 2026-10-24 | Founder scenarios (17 synthetic, author-written) | all | Rule base | Expectations met | 47 / 47 | `python -m startupiq.rules.evaluate` | Not independent: same author as the rules |
| 2026-10-24 | Scenarios + 80 real uploads (60 Rossmann, 20 Olist marketplace) | all | Rule base | Rules that fire at least once | 89% scenarios, 55% real | same | Unfired rules need inputs real uploads lack |
| 2026-10-24 | Olist share, test split, categories with 5+ sellers | test | Share LightGBM as an alert | Precision of "share drops by more than 2 points" | 0.69 (alerts on 14% of rows) | check in decision 0013 | Same precision at 0.5 points but 34% of rows alerted: threshold raised to 2 points |
| 2026-10-24 | Planner: 4 businesses x 3 targets (12 problems, 10 solvable) | all | BFS / DFS / Uniform-cost / **A\*** / Greedy | Optimal plans found (of 10) | 3 / 2 / 10 / **10** / 2 | `python -m startupiq.search` | A* matched uniform-cost cost in 12 of 12; mean nodes expanded 2,835 vs 3,176 |
| 2026-10-24 | Planner, medium targets | all | Uniform-cost vs A* | Nodes expanded | Olist 868 vs 159; Rossmann 463 vs 59; UCI 453 vs 9 | same | Little gain on the hardest targets (15,716 vs 14,793) |
| 2026-10-24 | Weekly panel, 2,539 series | validation | MLP / LSTM+attention / **TCN** (3 seeds each) | MASE pooled | 0.720 / 0.716 / **0.686 ± 0.009** | `python -m startupiq.models.run dl_weekly` | None beats Auto (0.616); all beat it on UCI products (TCN 0.600 vs 0.703) |
| 2026-10-24 | Weekly panel | validation | LSTM with vs without attention | MASE pooled | 0.695 vs 0.776 | same | Attention helps |
| 2026-10-24 | Rossmann daily | validation | MLP / LSTM+attention / **TCN** | RMSPE | 0.145 / 0.159 / **0.131 ± 0.003** | `... dl_rossmann` | Beats LightGBM 0.134, not the forest 0.119; coverage 0.82. First run 0.549 because all origins fell on one weekday; fixed |
| 2026-10-25 | Weekly panel | validation | **Hybrid: Auto + TCN candidate** | MASE pooled | **0.603 ± 0.002** vs Auto 0.616 | `... dl_hybrid` | Meets the decision 0005 rule; serves (decision 0015) |
| 2026-10-25 | 600 validation windows | validation | LSTM attention vs occlusion | Median Spearman | -0.11 (saliency vs occlusion 0.63) | `python -m startupiq.dl.interpret` | Attention is not a faithful explanation here |
| 2026-10-26 | 300 unseen Olist sellers | last 12 weeks | Product forecaster / naive / moving average | MASE | 0.982 / 1.036 / 0.936 | `python -m startupiq.evaluate.validation` | Coverage 0.56; small sellers 0.917, large 1.105; 0.26 s per analysis; deterministic 40/40 (before the serving TCN existed) |

Exploration numbers use raw data and will be re-measured after Step 2 cleaning; the re-measured rows supersede these.
