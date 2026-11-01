# Why deep learning for market behaviour forecasting (answer key)

Questions an examiner, investor or collaborator is likely to ask about Phase 5, with short answers first and the
reasoning after. The honest limits are included on purpose: they are part of a good answer.

---

## 1. What does deep learning do in StartupIQ?

**Short answer:** it forecasts **market behaviour**: where interest in a startup's sector is heading over the
next 4-12 weeks, as a range (P10 / P50 / P90). Market interest is measured from daily Wikipedia page views of each
sector's topics and brands (2023-2026, public domain).

Example of what the founder sees:

> Interest in your market (food and beverage brands) is expected to rise about 12% over the next 3 months, peaking in
> October. Over the last year your revenue moved with your market at about 0.8×.

It does **not** forecast one company's sales with deep learning. Everything that must be exact stays plain
calculation: P&L, EBITDA, burn, runway, cap table, dilution.

| Part | Method |
| --- | --- |
| Expense categorisation | ML (done) |
| Market interest forecast per sector | **Deep learning** |
| "You vs your market" comparison | Simple statistics on the startup's own months |
| Runway and cash | Exact calculation from the P&L |

---

## 2. Why forecast the market and not the startup's own sales?

**Short answer:** because the data for it exists, it is recent, and the test is fair.

- **No mismatch.** Startup monthly financials are private; no open dataset exists. Training on shop sales and
  predicting a startup would be comparing different things. Here we train on market interest and predict market
  interest: the same kind of data in and out.
- **Recent data exists.** Wikipedia page views are public (CC0), daily and current, so the model can be trained
  only on the post-COVID, AI-era market.
- **Directly testable.** Hide the last months of a market signal, forecast them, compare with what happened.
- **Useful to the founder.** "Is my market heating up or cooling down? When is the peak? Am I growing faster or
  slower than my market?" These decide hiring, stock and marketing timing.

---

## 3. Why only 2023-2026 data?

**Short answer:** markets changed after COVID, UPI and the AI boom. A model trained on 2011-2018 data (Walmart M5,
Rossmann, Olist) would learn a market that no longer exists.

The cost is fewer yearly cycles (about three). We make up for it by going **finer and wider instead of further
back**:

- **Finer:** daily data (about 1,400 days per article, about 200 weeks per sector) instead of monthly.
- **Wider:** 19 sectors and about 170 articles (topics and brands, English and Hindi), all from 2023-2026.

Honest limit: with only three festive seasons in the data, ranges for festive months are wider. The report says so.

---

## 3b. Why Wikipedia page views? Isn't that just "interest", not demand?

**Short answer:** yes, it measures interest, and we say so. It is the best free, recent, daily, openly licensed
signal of what people in a market are paying attention to.

- **Why not Google Trends:** its official API is a closed, invite-only alpha (2025-26) and the free library
  (pytrends) was archived in April 2025; scraping breaks Google's terms.
- **Why not payment or GST data:** NPCI UPI, GST and RBI card data are public but monthly (about 45 points for
  2023-2026), too few to train a deep model. They can be added later as extra monthly inputs.
- **Licence:** Wikimedia analytics data is released under CC0 (public domain).
- **How it is built:** each article's daily human views (bots excluded) summed over its title and every redirect
  (so renamed articles keep their history), scaled to its own median so big and small articles count equally,
  viral one-day spikes capped, then the median across a sector's articles. See ml/market/index.py.
- **The chatbot effect:** Wikipedia readership fell from 2023 to 2026, and topic pages fell fastest, because people
  now ask chatbots for factual lookups. Two corrections: (1) divide by all human views on that Wikipedia each day
  (share of attention), and (2) divide each sector by the median of all 19 sectors (100 = a typical market), so a
  decline shared by every sector cancels out. After this, AI tools rise sharply, fashion peaks every Diwali and
  edtech peaks in exam-results season, as expected. Interest in chatbot articles is also given to the models as an
  input, so they can separate "market cooling" from "readers moved to chatbots".
- **Limits:** Wikipedia readers lean English-speaking and urban; Hindi articles are included but fewer. Views are
  per language, not per state.

---

## 3c. What do the models have to beat?

Weekly relative index, 19 sectors, forecasting 1-12 weeks ahead. Split by time: train Jan 2023 - May 2026,
validation May - Aug 2026, test Aug - Oct 2026 (locked until the end). MASE below 1 = better than "no change".

| Baseline | Validation MASE | Test MASE | P10-P90 coverage (test) |
| --- | --- | --- | --- |
| Naive ("same as this week") | 1.85 | 1.50 | 70% |
| Seasonal naive ("same week last year") | 2.40 | 1.93 | 56% |
| Exponential smoothing, damped trend | 1.56 | 1.48 | 69% |

**Result (full detail in ml/market/RESULTS.md):** LSTM with attention scored test MASE 1.367 vs 1.477 for
exponential smoothing (7.4% better; 18% better on validation), with P10-P90 coverage of 91% vs 69%. The CNN and MLP
also beat every baseline. Per sector the LSTM wins 10 of 19, mostly the festival-driven ones (food and beverage
0.55 vs 1.26); smooth-trend sectors such as AI tools still favour exponential smoothing, which is why the app picks
the model per sector.

Exponential smoothing was the bar. Seasonal naive is worst: with only about three years, last year's pattern is a
noisy guide on its own, which is exactly the gap a model learning across sectors should close. All baselines give
ranges that are too narrow (coverage under 80%).

---

## 4. Why not rules? ("interest next week = interest last week + last month's growth")

**Short answer:** a rule must be written by someone who already knows the pattern, and patterns differ by sector
and change over time.

- Every sector has its own rhythm: D2C peaks at Diwali, edtech at admission season, B2B dips in March (budget
  year end), food delivery with monsoon and cricket. A rule per sector and region cannot be written by hand.
- Rules cannot say how sure they are. A founder needs the bad case, not only the middle one.
- Rules break silently when a market shifts; the model is re-checked against reality every month.

Where rules are the right tool, we use them: Indian bank formats, GST/TDS handling, health checks.

---

## 5. Why not a mathematical / statistical model (moving average, linear trend, ARIMA, exponential smoothing)?

**Short answer:** classic models fit each series alone and assume fixed shapes. With only three years of data,
one series alone is too short. A deep model learns from thousands of related series at once (every sector and
state), so patterns seen in one market help forecast another.

| | Statistical model (ARIMA, ETS, trend) | Deep learning (global model) |
| --- | --- | --- |
| Learns from | one series at a time | thousands of related series together |
| With 3 years of data | sees each yearly pattern ~3 times, unreliable | shares patterns across sectors and articles |
| Shape of patterns | fixed forms (straight trend, fixed season shape) | any shape: spikes, slowdowns, step changes |
| Extra inputs | hard to add (festivals, holidays, related signals) | easy to add as features |
| Uncertainty | from formula assumptions, often too narrow | learned directly as P10 / P50 / P90 |

We still include the statistical models as **baselines** and use them when they win (question 7).

---

## 6. Why these networks: MLP, LSTM, 1D CNN?

Each makes a different assumption about how time works, so comparing them is meaningful:

- **MLP:** sees the last N weeks as a flat list. Simple, fast, a strong baseline.
- **LSTM with attention:** reads weeks in order with a memory; attention shows which past weeks mattered, which
  we can display to the founder.
- **1D CNN (temporal convolution):** finds repeating local shapes (a month-end spike, a festive ramp-up).
  Trains faster than an LSTM and is often as accurate.

All three output P10 / P50 / P90 per future period using quantile (pinball) loss.

---

## 7. So is deep learning always better than statistics?

**No, and we say so.** Large forecasting competitions (M4, M5) showed simple methods are hard to beat on a single
series, while models trained across many related series did best when many were available.

So StartupIQ never assumes deep learning wins:

1. Every model, including baselines (naive, seasonal naive, exponential smoothing), is backtested on the same
   hidden recent period.
2. The best model per sector is used.
3. The comparison is shown.

---

## 8. How does the market forecast relate to the startup?

- The startup picks its sector at signup (already in the app).
- Its own monthly revenue comes from its bank statements (already built).
- Simple statistics measure how its revenue moves with its market (for example 0.8× or 1.5×) and whether it is
  beating the market.
- A simple projection of the startup's own revenue = its own trend adjusted by the market forecast. This is
  labelled as an estimate, not a deep learning prediction.

---

## 9. How do you know the forecast is any good?

- **Backtesting:** hide the last weeks or months of 2026, forecast them, compare with what happened. Measured with
  MAE / MAPE for the middle forecast and coverage for the range (actuals should fall between P10 and P90 about
  80% of the time).
- **Same test for all models**, same time split, so the comparison is fair.
- **Track record:** every forecast is saved and compared with what happened next month.

---

## 10. One-line answers for a viva

- *What does DL do?* Forecasts where market interest in each sector is heading, with a range, from Wikipedia page views.
- *Why the market and not the startup's sales?* Startup financials are private; market data is public, recent, and
  lets us train and test on the same kind of data.
- *Why only 2023-2026?* The market changed after COVID, UPI and AI; we use daily data across many sectors and
  articles to get enough data from three years.
- *Why not rules?* Every sector has its own rhythm, and rules give no confidence range.
- *Why not ARIMA/ETS?* They learn from one short series and assume fixed shapes; we keep them as baselines.
- *Is DL always better?* No: every model is backtested and the best one is used per sector.
- *What stays non-ML?* P&L, EBITDA, burn, runway, cap table and dilution, because they must be exact.
