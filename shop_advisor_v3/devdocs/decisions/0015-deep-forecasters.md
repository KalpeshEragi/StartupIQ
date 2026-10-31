# 0015: Deep forecasters (Task 7) and the hybrid that serves

- Status: Accepted. Choices written on 2026-10-25 from validation only, before any deep-model test run; test results appended below.
- Module / task: M3; Tasks 7, 8
- Code: `src/startupiq/dl/` (windows, models, train, experiment_weekly, experiment_rossmann, hybrid, interpret)
- Results: `docs/results/dl_weekly.md`, `dl_rossmann.md`, `dl_hybrid.md`, `dl_interpret.md` (+ `_test` versions)

## What was built

| Model | Role | Why this design |
| --- | --- | --- |
| MLP on the flattened window | The basic neural model (Task 8 comparison) | No notion of order: shows what sequence models add |
| LSTM with attention | Recurrent model the syllabus names; attention weights for Task 8 | Attention over the 52 (or 84) encoder steps lets each forecast step look at specific past weeks |
| TCN (1D-CNN) | Convolutional model the syllabus names; feature maps for Task 8 | Causal dilated convolutions (dilations 1..32, receptive field 127 steps) see a full year with few parameters |

All three: P10/P50/P90 heads that cannot cross (P50 plus two softplus offsets), pinball loss, AdamW, gradient clipping, learning-rate halving on plateau, early stopping on 10% of series held out of training. Inputs use only data before the origin (leakage checklist W1-W4; `tests/test_dl.py` checks the window builder and that the TCN is causal). Rossmann daily adds the planned calendar (open, promo, holidays), last year's same day, static store attributes and a store embedding, and masks closed days out of the loss.

## Compute

PyTorch 2.9.1 for CUDA 12.8 on the RTX 2050 (4 GB). The first attempt ran on the CPU-only PyTorch build and needed 2.5 hours for one LSTM; it was stopped and the GPU build installed (about 25x faster). Running three GPU jobs at once exhausted Windows' page file; jobs now run one at a time, with smaller evaluation batches and the GPU cache freed after each model.

## Validation results

**Weekly panel (pooled MASE; lower is better; 3 seeds for the best of 5 random-search settings per architecture):**

| Model | Rossmann stores | UCI products | Pooled | P10-P90 coverage |
| --- | --- | --- | --- | --- |
| Auto (Step 5, served) | **0.467** | 0.703 | 0.616 | 0.946 / 0.587 |
| MLP | 0.895 ± 0.057 | 0.617 ± 0.009 | 0.720 ± 0.026 | 0.780 |
| LSTM + attention | 0.901 ± 0.039 | 0.607 ± 0.002 | 0.716 ± 0.016 | 0.603 ± 0.138 |
| TCN | 0.834 ± 0.025 | **0.600 ± 0.002** | 0.686 ± 0.009 | 0.748 |
| **Hybrid: Auto with the TCN as a candidate** | 0.467 | 0.682 ± 0.004 | **0.603 ± 0.002** | **0.746** |

- On its own no deep model replaces Auto: the rule (beat the bar by more than one sd, coverage 0.70-0.90) fails because seasonal methods are far better on stores with a stable year.
- On sparse products every deep model beats every Step 5 method (TCN 0.600 vs Auto 0.703, LightGBM 0.620, ETS 0.637): the place Step 5 found Auto weakest.
- **Attention matters:** LSTM 0.695 with attention vs 0.776 without (same settings, seed 0).
- **Regularisation:** dropout 0.3 barely changes MASE (0.697 -> 0.707) but lifts coverage from 0.74 to 0.83; weight decay 1e-3 changes nothing. Holdout loss stays at or below training loss: the models are not overfitting.

**Rossmann daily (RMSPE on the same 37,472 store-days as Step 5; settings fixed in advance, 3 seeds):**

| Model | RMSPE | Coverage |
| --- | --- | --- |
| Random forest (Step 5) | **0.119** | n/a |
| TCN | 0.131 ± 0.003 | 0.820 |
| LightGBM (Step 5) | 0.134 | 0.767 (quantile model) |
| MLP | 0.145 ± 0.004 | 0.852 |
| LSTM + attention | 0.159 ± 0.002 | 0.845 |

The TCN beats LightGBM, ridge and every baseline but not the forest; it stays an experiment there. Its intervals are the best calibrated of any model so far.

## A bug worth remembering

The first Rossmann run scored RMSPE 0.549 (worse than the store mean). Training windows were cut every **14** days, so every training origin fell on the same weekday (Tuesday): the model learned the weekday from position in the window instead of from the weekday features, and at the Saturday validation origin every forecast was shifted by four days (30-60% too high on every weekday). Found by scoring origins day by day: 2015-05-05 was fine, 2015-05-09 was not. Fix: one origin every **10** days (10 and 7 share no factor, so origins cycle through all weekdays); `test_rossmann_training_origins_cover_every_weekday` guards it. Result: 0.549 -> 0.145 for the same MLP.

## Decision

1. **The product's weekly forecast becomes the hybrid**: Auto's per-series backtest on the last 12 weeks, with the TCN as a sixth candidate. It meets the decision 0005 rule on validation (0.603 ± 0.002 vs 0.616; coverage 0.746). It changes nothing for store-like series (the TCN is picked for 1%) and helps sparse ones (picked for 18%).
2. In the product the TCN is trained once on the whole weekly panel. A founder's data was never in that training set, so the same model can be backtested on the founder's last 12 weeks without leakage. The TCN needs 52 weeks of lookback plus 12 for the backtest, so it is only a candidate for uploads with 64+ weeks; shorter uploads keep Auto.
3. Rossmann daily: the random forest remains the best model; the TCN is reported as a tuned, interpreted experiment.

## Test results (scored once, after the choices above)

**Weekly panel, test origin (3 seeds, `docs/results/dl_weekly_test.md`):**

| Model | Pooled | Rossmann stores | UCI products | Coverage |
| --- | --- | --- | --- | --- |
| Auto (Step 5) | **0.810** | 0.448 | 1.020 | 0.695 / 0.536 |
| MLP | 0.927 ± 0.024 | 0.926 | 0.927 | 0.839 |
| LSTM + attention | 0.914 ± 0.038 | 0.856 | 0.948 | 0.704 |
| TCN | 0.861 ± 0.017 | 0.781 | **0.908** | 0.747 |

The validation picture holds on test: no deep model beats Auto overall, and on sparse products every deep model beats it (TCN 0.908 vs 1.020). The Rossmann-daily and hybrid test runs were interrupted twice by session ends and are not yet scored; until they are, the hybrid's evidence is validation only, and the serving TCN file is not yet trained, so the product falls back to plain Auto.
