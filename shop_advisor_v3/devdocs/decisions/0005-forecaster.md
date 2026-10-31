# 0005: Forecast model

- Status: Proposed. Classical half decided in Step 5 (decision 0012): the bar for the deep models is the **random forest**, RMSPE 0.119 validation / 0.130 test on Rossmann daily, and **Auto** (pooled MASE 0.616) on the weekly panel. Deep-model verdict after Step 8.
- Date: 2026-10-24
- Module / task: M3; Tasks 6, 7, 8

## Problem

Forecast 12 periods ahead with a P10 to P90 band, for series from 8 weeks to 3 years long, including businesses unlike the training data. The bar to beat today is seasonal naive: RMSPE 0.160 on Rossmann (last 42 days).

## Candidates

| Family | Model | For | Against | Role |
| --- | --- | --- | --- | --- |
| Baseline | Store mean, same weekday last 4 weeks, same day last year | Free, robust, honest floor | Ignores promos and trend | Floor every model must beat |
| Statistical | ETS / ARIMA per series | Classic, interpretable, good on short series | One model per series; weak on promos and cross-series patterns | Classical comparison; fallback for short uploads |
| Classical ML | **LightGBM on lag, calendar and promo features** | Usually the strongest classical model for retail sales (it dominated the M5 competition); fast; feature importance | Needs hand-made features; forecasts each step separately or recursively | Expected main benchmark; in Step 5 the random forest beat it on validation (0.119 vs 0.134) and they tied on test (0.130 vs 0.128), see 0012 |
| Classical ML | Random forest, ridge | Simple, robust | Random forest cannot extrapolate trend; ridge underfits interactions | Extra rows in the comparison |
| DL | **LSTM (global, seq2seq, quantile loss)** | Learns sequence patterns across all series; one model for all businesses; native multi-step output; the RNN family the syllabus asks for | Needs scaling per series and care with short series; slower | Primary DL model (Task 7) |
| DL | **1D-CNN / TCN** | Learns local patterns such as weekly cycles and promo spikes; fast; filters can be visualised (Task 8 feature maps) | Fixed receptive field | Second DL architecture |
| DL | LSTM with attention | Attention weights show which past weeks mattered | Attention is not always a faithful explanation | Task 8 interpretability comparison |
| DL | GRU | Fewer parameters than LSTM, often similar accuracy | Very similar to LSTM, little to learn from adding it | Optional ablation only |
| DL | Transformer / Temporal Fusion Transformer / N-BEATS / DeepAR | State of the art on large benchmarks | Heavier, many hyperparameters, little advantage at about 1,000 series; TFT and N-BEATS are not in the syllabus list | Not used; mention as future work |
| Pretrained | Foundation time-series models (e.g. Chronos) | Strong zero-shot | Not something we built or can interpret; does not demonstrate Task 7 | Not used |

## Criteria, in order (fixed now)

1. Validation backtest error (RMSPE for Rossmann, MASE pooled across datasets), mean of 3 seeds.
2. P10 to P90 coverage within 70 to 90% on validation.
3. Leave-one-dataset-out error (generalisation to a new business).
4. Inference time under 1 second per upload on CPU.
5. Interpretability (what we can show the founder).

## Decision rule (written before training)

- The DL model serves forecasts in the product only if its validation error is lower than the best classical model's (random forest on Rossmann daily; Auto on the weekly panel) by more than one standard deviation across seeds **and** its coverage passes criterion 2. Otherwise the classical model serves, and the DL model is reported as a tuned, interpreted experiment.
- If two models tie (within one standard deviation), the simpler and faster one serves.
- For uploads below the Standard tier, ETS or seasonal naive serves regardless.

## Evidence

Classical results: decision 0012 and `../evaluation_log.md` (2026-10-24 rows). Deep-model results: after Step 8.
