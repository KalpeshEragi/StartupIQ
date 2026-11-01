# Market interest forecasting: results

Run: `python -m market.evaluate` then `python -m market.train` (from `ml/`). GPU: NVIDIA RTX 2050, about 7 minutes.
Raw numbers: `ml/models/runs/market_baselines.json`, `ml/models/runs/market_dl.json` (not committed).

## Setup

- **Target:** weekly relative market-interest index for 19 sectors (100 = a typical market), built from daily
  Wikipedia page views (CC0), Jan 2023 to Oct 2026. Corrected for the chatbot effect (share of all Wikipedia views,
  then relative to the median of all sectors).
- **Task:** forecast weeks 1-12 ahead as P10 / P50 / P90.
- **Split by time:** train Jan 2023 - May 2026 (175 weeks), validation May - Aug 2026 (12 weeks), test Aug - Oct
  2026 (12 weeks). Test was scored once, after all choices were made on validation.
- **Training data:** one global model over 189 series (19 sectors + about 170 articles), 20,638 windows for tuning,
  22,906 for the final fit (train + validation). Inputs: 52 past weeks, chatbot-interest change, week of year,
  Diwali and Holi weeks, sector embedding.
- **Tuning:** dropout {0.1, 0.3} x epochs {10, 20, 40, 60}, 3 seeds each, seed forecasts averaged. Every model chose
  dropout 0.3, 10 epochs (longer training overfit the short history).
- **Metric:** MASE = mean absolute error / average week-to-week change in training. Multi-week forecasts naturally
  score above 1 (errors grow with the horizon); what matters is the comparison on the same scale. Coverage =
  share of actual weeks inside P10-P90 (target about 80%).

## Results (19 sectors, 1-12 weeks ahead)

| Model | Validation MASE | Test MASE | 1-4 weeks | 5-8 weeks | 9-12 weeks | P10-P90 coverage (test) |
| --- | --- | --- | --- | --- | --- | --- |
| **LSTM + attention** | 1.283 | **1.367** | **1.17** | **1.53** | 1.40 | 91% |
| 1D CNN (TCN) | **1.281** | 1.405 | 1.27 | 1.57 | **1.37** | 86% |
| MLP | 1.355 | 1.436 | 1.32 | 1.58 | 1.41 | 95% |
| Ensemble (mean of 3) | 1.281 | 1.385 | 1.24 | 1.54 | 1.37 | 90% |
| Exponential smoothing (best baseline) | 1.562 | 1.477 | 1.23 | 1.80 | 1.40 | 69% |
| Naive | 1.846 | 1.502 | 1.24 | 1.85 | 1.42 | 70% |
| Seasonal naive | 2.402 | 1.934 | 1.92 | 2.03 | 1.85 | 56% |

## What this shows

- **All three networks beat every baseline** on validation (18% better than exponential smoothing) and on the
  locked test (LSTM 7.4% better).
- **The gain is largest 5-8 weeks ahead** (1.53 vs 1.80), where pattern knowledge matters more than "last value".
- **Ranges are honest:** baselines' P10-P90 caught only 69-70% of actual weeks (too narrow); the networks caught
  86-95% (slightly wide; the CNN at 86% is closest to the 80% target).
- **Per sector, LSTM beats exponential smoothing in 10 of 19.** Biggest wins: food and beverage (0.55 vs 1.26),
  food delivery (1.41 vs 2.92), logistics (0.66 vs 0.99), edtech (1.38 vs 1.76), media (1.89 vs 2.28). The test
  window (Aug - Oct 2026) runs into Diwali, and these are festival-driven sectors. Biggest losses: AI tools (2.43 vs
  1.99) and real estate (1.14 vs 0.79), where a smooth trend is hard to beat.
- **So the app should not use one model everywhere:** per sector, pick the model by its validation error, and show
  the comparison. This is the plan stated in docs/WHY_DEEP_LEARNING.md, now backed by numbers.

## Limits

- About three yearly cycles of data (2023-2026 by design), so festival effects are learned from three examples.
- 10 epochs was the smallest setting tried and won for every model; a finer search below 10 may help.
- The index measures attention on Wikipedia, not sales. It is a market-interest signal, labelled as such.
