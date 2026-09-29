# Task 5 — Feature Engineering, Leakage Analysis, Selection

**Run:** `python Task5/run_task5.py` (needs Task 4 output). Code: `src/features/build.py`, `src/features/select.py`.

**Modelling population:** the 6,295 resolved companies (acquired 3,692 / closed 2,603). `operating` (41,827) is unresolved and `status`-missing (1,314) has no label, so both are excluded from supervised training. Target: `1 = acquired`, `0 = closed`.
**Split:** 80/20 stratified, `random_state=42` → train 5,036 / test 1,259 (acquired share 58.7% / 58.6%). Saved with a `split` column in `data/processed/model_frame.csv`. The test set is untouched by selection and preprocessing.

## Feature sets (defined by leakage risk)
| Set | Contents | Role |
|---|---|---|
| `profile_only` | market, country, region, n_categories, age at first funding | ablation |
| **`early_safe`** | profile + seed/angel/convertible/grant/crowdfunding signals | **primary, prediction-safe** |
| `early_safe_plus_calendar` | early_safe + founded_year, first_funding_year | ablation (cohort effect) |
| `full_history` | early_safe + total funding, rounds, venture, last-funding timing, etc. | leaky upper bound for comparison only |

## Outputs
- `Task5/outputs/feature_registry.csv` — every feature: tier, source columns, calculation, reason, availability at evaluation time
- `Task5/outputs/selected_features.json`, `selection_dropped_*.csv`, `importance_*.csv`
- `models/preprocessor_{early_safe,full_history}.joblib` — preprocessors fitted on train only
- `data/processed/X_{set}_{train,test}.csv` — encoded, ML-ready matrices (55 columns early_safe, 59 full_history)

Details: `task5_feature_engineering.md`, `task5_feature_selection.md`.
