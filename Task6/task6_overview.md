# Task 6 — Machine Learning

**Run:** `python Task6/run_task6.py` (needs Task 5 output). Code: `src/models/train.py`, `src/evaluation/metrics.py`. Outputs: `Task6/outputs/`, models in `models/`.

## Target
**Acquired (1) vs Closed (0)** among the 6,295 companies with a resolved outcome (58.7% acquired). Verified against the data: `status` has operating 41,827 / acquired 3,692 / closed 2,603 / missing 1,314 and no IPO class. `operating` is unresolved (the company may still be acquired or close later), so training on it as a "negative" class would mislabel companies that simply haven't finished their story. Missing-status rows have no label.
This is a *success vs failure of resolved outcomes* model. It is not INVEST/INCUBATE/REJECT; those come from the future decision engine.

## Setup
- Train 5,036 / test 1,259 (stratified, fixed in Task 5). Test set used once per model for reporting; model choice is by 5-fold CV ROC-AUC on train.
- Preprocessing sits inside each pipeline, so it is re-fitted on the training part of every CV fold.
- Imbalance: mild (59/41), so `class_weight="balanced"` only; no resampling.
- Models: logistic regression (baseline), random forest, histogram gradient boosting (only on the two headline feature sets). Details: `task6_models.md`; results and error analysis: `task6_evaluation.md`.

## Saved artifacts
- `models/outcome_model_early_safe.joblib` (+ `_features.json`): primary, prediction-safe model (HistGradientBoosting; a near-tie with random forest by CV)
- `models/outcome_model_full_history.joblib`: leaky comparison model
- `models/preprocessor_*.joblib` (Task 5)
