# Ethics, fairness and responsible use (Task 10)

## What the system must never do, and how it is enforced

| Risk | Safeguard | Where |
| --- | --- | --- |
| Recommending illegal conduct (agreeing prices with rivals) | The move exists in the ontology only so it can be refused: constraint `legal`, rule R90 (forbid, priority 100), planner excludes it | `configs/moves.yaml`, `configs/rules.yaml`, `search/problem.py` |
| Confident output from thin data | Data tiers; refusal below 20 rows / 4 periods; "add your finances" instead of a recommendation when cash data is missing (R84) | decision 0004, 0013 |
| Presenting estimates as facts | Every card carries its source: measured, predicted, inferred; rival facts are labelled "estimates, not observations"; rival price cuts are stated as unpredictable | `contracts.Source`, advisor cards |
| Overstating certainty | Forecasts shown as a range; rule conclusions carry certainty factors derived from measured model quality; the range is called "likely", not "80%", because measured coverage is below 80% | decisions 0012, 0013 |
| Bad advice under cash stress | Forbid rules (no new product or channel while runway < 3 months); constraint checks on every recommendation and every planner action | R13, knowledge constraints |
| Surveillance of competitors | Only public signals or what the founder states; no scraping | design, decision 0002 |
| Privacy of uploads | Files processed in memory, never written to disk or logged | `api/app.py` |

## Fairness and bias found

- **Seller size.** On 300 unseen Olist sellers the forecaster is relatively better for small sellers (MASE 0.917) than large ones (1.105); the competitor classifier is less accurate for small sellers (accuracy 0.50 vs 0.68). Advice for small sellers therefore rests more on rules and the founder's own numbers than on rival predictions (`docs/results/validation.md`, `competitor.md`).
- **Survivorship.** Olist and YC-style data over-represent sellers that kept trading; failure patterns are under-sampled.
- **Geography.** Training data is German (Rossmann), UK (UCI) and Brazilian (Olist); market signals are US (FRED). Indian small sellers, the intended users, are not in the training data. The pilot with real local businesses (Step 12 plan) is the remedy; until then outputs are labelled as estimates.
- **Assumed effects.** Demand lifts and costs of moves are assumptions (`configs/moves.yaml`), except the promotion lift, which is grounded in Rossmann data. The planner's plans are starting points, not promises.

## Licences

Lab and academic use only: Olist is CC BY-NC-SA 4.0 (non-commercial); Rossmann is under Kaggle competition rules; UCI is CC BY 4.0. A commercial product would need its own or licensed data (decision 0002).

## Limits a founder should know (shown in the app or the report)

1. The system cannot see competitors' sales; rival outputs are inferences.
2. Forecast ranges are too narrow on unseen businesses (56% coverage instead of 80%).
3. It is decision support. The founder makes the call.
