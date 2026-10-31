# 0006: Competitor-move classifier

- Status: Accepted (2026-10-24). Random forest chosen; test macro-F1 0.366 vs 0.228 baseline. Evidence and product consequences: decision 0012, section 3.
- Date: 2026-10-24
- Module / task: M7; Tasks 5, 6

## Problem

Predict what a seller in a category does next month: price cut, expansion, stagnation or exit. Trained on Olist, where all sellers are observed. Applied to a founder's competitors, where it is an inference.

## Label definitions (fixed before modelling)

Computed per seller-category-month at month t, for sellers with orders at t and at least 2 active months up to t (as implemented in `features/olist.py`; the draft filter "active 6+ months overall" used future months and was dropped, see leakage checklist O4):

| Label | Definition | Fixes from exploration |
| --- | --- | --- |
| price_cut | Same-product median price falls by more than 5% next month (products sold in both months) | Exploration used all-item median price, which mixes product changes into price |
| expansion | Orders up more than 30% and at least 3 orders next month | — |
| exit | No orders in t+1, t+2 and t+3 | Exploration counted any empty month as exit. The draft's "and none again in the data" was dropped because it looks past the split boundary (leakage checklist O3) |
| stagnate | Everything else | — |

Draft class shares before the fixes: stagnate 39%, price cut 25%, no orders next 21%, expansion 15%.

Final class shares (training split): stagnate 63%, expansion 19%, exit 16%, price cut 2%. The same-product price rule shows that most of the draft's "price cuts" were product-mix changes; genuine price cuts are rare and, with these features, unpredictable (F1 0.00).

## Candidates

| Model | For | Against |
| --- | --- | --- |
| Majority class | Floor | — |
| Logistic regression (multinomial, class weights) | Coefficients readable; calibrated probabilities | Linear only |
| Linear and RBF SVM | Strong margins on moderate data | No native probabilities (needs Platt scaling); slow RBF at 13k rows is fine but tuning is fiddly |
| Decision tree | Rules can be printed and fed to the rule engine | High variance |
| Random forest | Robust, little tuning | Probabilities less calibrated |
| Gradient boosting (LightGBM) | Usually best on tabular data of this size | Needs tuning; calibrate probabilities |
| Voting ensemble of the above | Can add a little | Harder to explain |
| MLP | DL option | 13k tabular rows: rarely beats boosting; not worth the complexity |

## Criteria

1. Macro-F1 on the grouped temporal validation split (classes are imbalanced, so accuracy would reward always predicting stagnate).
2. Calibration (Brier score), because probabilities feed the rule engine as certainties.
3. Explainability of the top features.

## Decision rule

Pick the best macro-F1 within one standard deviation; among ties pick the better calibrated one; among further ties the simpler one.
