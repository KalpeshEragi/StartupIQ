# Demo script (about 12 minutes)

Three parts: one ML application, one deep-learning forecast, one rule-based safety gate. Show the evidence in the code first, then the website.

## Before you start

```bash
cd startupiq_v2
python -m pytest -q                          # 242 passed
uvicorn startupiq.api.app:app --port 8010    # open http://127.0.0.1:8010
```

If you retrain from scratch: `python -m startupiq.dl.serve_train` (deep model the website uses, about 4 minutes on the GPU), then `python -m startupiq.segment` (business profiles, about 1 minute).

## Part 1: the problem (1 minute)

> "A small shop owner has a sales sheet and two questions: how much will I sell in the next three months, and which moves are safe for my cash? Startup IQ answers in three steps. First, it works out what kind of business this is. Then it forecasts with the model that predicts this shop's own history best. Then rules check every suggestion against cash, margin and stock."

## Part 2: show the evidence in the code (5 minutes)

### Step 1, ML: business profiling (`src/startupiq/segment.py`, `docs/results/segments.md`)

- Open `profile()`. "Each sales history becomes seven numbers that don't depend on currency or shop size: volatility, weeks with no sales, peak vs a normal week, how seasonal it is, half-year trend, how much one week predicts the next, and typical weekly swing. The same shop in rupees or pounds gets the same profile; `tests/test_segment.py` checks that."
- Open `segments.md`. "K-means on 2,539 real weekly series (Rossmann stores, UCI online-shop products). We tried k = 3 to 6 and picked k = 4: highest silhouette (0.45) among stable solutions (ARI 0.996 across seeds)."
- The four groups: Steady, Moderate, Intermittent, Declining sellers.
- **The key slide**: the benchmark per group. "On steady sellers, statistics beat the deep model clearly (MASE 0.47 vs 0.83). On the other three groups, the deep model wins on 55 to 59% of series. So the kind of business decides which kind of model to trust."

### Step 2, deep learning: the forecast (`src/startupiq/dl/models.py`, `docs/results/dl_weekly.md`, `dl_weekly_test.md`, `dl_hybrid.md`)

- Open `dl/models.py`. "Three architectures, all predicting a range (P10, P50, P90), not one number: an MLP, an LSTM with attention, and a 1D CNN, which is a temporal convolutional network with dilated causal convolutions."
- Open `dl_weekly_test.md`. "Tuned on a validation window, then scored once on a held-out test window. MASE, lower is better:"

  | Model | Validation | Test |
  | --- | --- | --- |
  | MLP | 0.720 | 0.927 |
  | LSTM + attention | 0.716 | 0.914 |
  | **1D CNN (TCN)** | **0.686** | **0.861** |
  | Statistical Auto (reference) | 0.616 | 0.810 |

- Say this openly: "The 1D CNN won among neural networks on both windows. On its own it doesn't beat statistics across all series, so we don't use it blindly. It enters a contest on each shop's own last 12 weeks, and the winner forecasts the next 12. That hybrid scored 0.603 vs 0.616 for statistics alone (`dl_hybrid.md`)."
- `dl/serve_train.py` trains the final 1D CNN that the website loads.

### Step 3, rules: the safety gate (`configs/rules.yaml`, `src/startupiq/advisor.py`)

- "47 forward-chaining rules turn the forecast and the founder's money figures into warnings and suggested moves. Every suggested move is then checked against constraints such as the margin floor, minimum cash and stock lead time. A move that breaks one is blocked, with the reason shown."
- "Each rule has an example and a counter-example in `tests/test_rules.py`: 47/47 fire on their example and stay silent on their counter-example."

## Part 3: the website (5 minutes)

1. **Healthy growing shop** sample (fills in its money figures). Click Analyse.
   - Headline and KPI tiles: "About ₹11.9 lakh over 12 weeks, up 39%, busiest week 2 November."
   - Forecast chart: "Grey is the last year, green is the forecast, and the shaded band is the likely range."
   - **Business profile**: "Placed as a Moderate seller. The bars compare this shop with the typical member of its group. It's flagged unusual because this is a synthetic demo shop, and the system says so rather than pretending."
   - **Model contest**: "Every model forecast the last 12 weeks without seeing them. Seasonal naive missed by 3.7%, the 1D CNN by 12.8%. Statistics won here, which matches what we measured: steady, seasonal shops suit statistics."
   - **Rules / What to do next**: warnings with reasons, the suggested move, and anything the safety gate blocked.
   - **How the deep model was chosen**: the MLP / LSTM / 1D CNN table, live from the result files.
2. **Real online-shop product (unseen by the model)**. "Real UCI data for a product the network was never trained on. It's an intermittent seller. Here the 1D CNN wins the contest: 34.7% error vs 72.1% for the best statistical method, and the forecast comes from the neural network."
3. Optional: **Growing but cash crunch** to show the safety gate blocking moves.

## Questions you are likely to get

- **"Is the 1D CNN the TCN?"** Yes. A TCN is a 1D convolutional network with causal, dilated convolutions, so each prediction only sees the past.
- **"Why didn't deep learning win on the first shop?"** Because statistics really were better on that shop's last 12 weeks. The system picks per shop and reports the result, so it never forces the neural network in.
- **"What does MASE mean?"** Forecast error divided by the error of repeating last year's same weeks. Below 1 is better than that baseline.
- **"Does the profile choose the model?"** Today it explains which kind of model tends to win for businesses like this one, and the contest decides. Routing the model by profile is the next step, and needs testing on held-out data first.
- **"Is my file stored?"** No. It is processed in memory and never written to disk.
