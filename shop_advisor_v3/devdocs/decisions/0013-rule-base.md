# 0013: Rule base, facts from models, and the safety gate

- Status: Accepted (2026-10-24)
- Module / task: M8 diagnosis, M10 safety; Tasks 1, 2, 9
- Code: `configs/rules.yaml`, `src/startupiq/rules/` (engine, loader, facts, evaluate), `src/startupiq/advisor.py`

## Problem

Turn numbers (vitals, forecast, Step 5 model outputs) into advice a founder can act on and argue with, without hiding uncertainty, and without ever suggesting something unsafe or illegal.

## Design

**Facts.** `rules/facts.py` is the only place model outputs become rule inputs. Each fact has a source and a certainty taken from measured quality where we have it:

| Fact | From | Certainty | Why that number |
| --- | --- | --- | --- |
| growth_recent, runway_months, stock_cover_weeks, sales_volatility, top_product_share | arithmetic on the upload (M1) | 1.0 (stock cover 0.7 when margin is defaulted) | measured |
| forecast_growth, weeks_to_peak, peak_ratio | Auto forecaster (M3) | 1 - band width / 2, clipped to 0.3..0.95 | a wide forecast range means a weak claim |
| market_pulse | "same weeks last year" rule (M5, decision 0012) | 0.60 (recent-slope fallback 0.45) | test macro-F1 0.62 / 0.48 |
| rival_p_exit, rival_p_expansion (marketplace mode) | competitor random forest (M7) | 0.55 | test macro-F1 0.37, Brier 0.56: better than chance, far from sure |
| rival_p_* (named rivals, no data) | Olist base rates 16% / 19% / 2% | 0.30 | a base rate says little about a particular rival |
| share_change_expected | share LightGBM (M7) | 0.69 | 69% of predicted drops over 2 points were real drops on test |

**Rules.** 47 rules in 11 groups (data, cash, stock, demand, market, rivals, dependence, diagnosis, recommend, safety, pre-mortem), stored as data in `configs/rules.yaml`, each with a firing and a non-firing example. Conclusion kinds: `assert` (a derived fact such as `cash_risk`), `warn`, `diagnose` (several can hold at once), `recommend`, `forbid`, `risk` (pre-mortem).

**Inference.** Forward chaining with conflict resolution by priority, then specificity, then file order; refraction (each rule fires once per run); certainty = rule cf x weakest condition; MYCIN combination for repeated conclusions; facts below certainty 0.2 count as unknown. Backward chaining answers "why not X?" by walking from `recommend:X` to the conditions that fail, one level into derived facts, and distinguishes "false" from "not given".

**Safety gate (M10).** Three layers, strongest first: (1) `forbid` beats `recommend` inside the rule base (e.g. R13: no new product while cash is short; R90: never price coordination); (2) every surviving recommendation is checked against the knowledge model's constraints (`KnowledgeBase.check`), and a move that breaks one is blocked with the reason; (3) R84: when finances are missing, no move is recommended at all.

## Thresholds and why

| Threshold | Value | Reason |
| --- | --- | --- |
| Runway critical / tight | < 3 / 3-6 months | 3 months is the time to cut costs or raise money; 6 months is the payback of a channel test (constraint `runway_at_least_6_months`) |
| Short history | < 26 weeks | the Standard tier boundary (decision 0004) |
| Wide range | band > 60% of the middle | above this the "likely range" spans more than half the expected value either way |
| Sales rising / falling, forecast up / down | +-10% | on Rossmann stores the change between consecutive 4-week totals has a median of 4.6% and a 75th percentile of 8.6%; only 20% of changes exceed 10%, so +-10% flags the unusual fifth, not routine noise |
| Busy season ahead | peak >= 1.25x recent median, within 8 weeks | 8 weeks is two supplier cycles at a 4-week lead time |
| Rival likely to expand / exit | > 35% | about twice the base rates (19% / 16%) |
| Share slipping | predicted drop > 2 points | about the share model's test error (MAE 1.55); see "Changes after real data" |
| Overstock | > 12 weeks of cover with falling demand | a quarter of sales sitting on shelves |
| Price war forbids matching | margin < 30% | a 10% cut would leave under 20% |

## Evaluation (python -m startupiq.rules.evaluate; docs/results/rules.md)

- Every rule fires on its example and stays silent on its counter-example (tests/test_rules.py).
- Founder scenarios: 17 synthetic situations, 47 written expectations, **47 met**. Caveat: written by the same author as the rules, so this measures consistency with stated intent, not independent expert agreement. An independent check is part of the pilot in Step 12.
- Coverage: 89% of rules fire somewhere in the scenarios; 55% fire on 80 real uploads (60 Rossmann stores with stated assumed finances, 20 Olist sellers in marketplace mode). Rules that never fired on real data are the ones whose inputs real uploads did not include (founder-reported rival price cuts, margin below floor, overstock).

## Changes after the first real-data run

1. **Share alerts were noise.** R43 used a 0.5-point drop; it diagnosed "losing share" for almost every top Olist seller. A check on the test split showed the model is not biased (large sellers really lose 1.8 points a month on average) and its alerts are right about 69% of the time at any threshold, so the threshold moved to 2 points (about the model's error): alerts fall from 34% to 14% of seller-months at the same precision.
2. **"Keep things as they are" while a problem was visible.** R83 recommended holding whenever no rule recommended anything, including sellers diagnosed as losing ground but with no finances given. R83 now requires finances and no negative diagnosis; the new R84 says a recommendation needs the founder's cash, costs and margin.
3. Informational messages (add cash, add stock, rival price cuts are unpredictable) were counted as warnings; they now go to an "add to unlock" list and the rivals card note.

## Limits

- Thresholds are judgement, written down to be argued with; the pilot should revisit them.
- Effects of moves (demand lift, cost) are assumptions in `configs/moves.yaml`, flagged with their source; only the promotion lift is grounded in data (Rossmann 1.39x).
- Rival facts outside marketplace mode are base rates, shown as estimates.
