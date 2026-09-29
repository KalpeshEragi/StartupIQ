# Task 5 — Feature selection and preprocessing

All steps use the training split only (5,036 rows).

**Procedure (`src/features/select.py`):** (1) drop >60% missing; (2) drop near-constant (>99.5% one value); (3) for numeric pairs with |Spearman| > 0.9 keep the one more associated with the target; (4) permutation importance (ROC-AUC) of a random forest on a 25% validation slice of the training data; drop features below 0.001 AUC.

| Set | Candidates → selected | Dropped |
|---|---|---|
| profile_only | 5 → 5 | none |
| **early_safe** | 12 → **7** | crowdfunding (99.9% zero), log_seed & log_angel (redundant with has_seed/has_angel), convertible note, grant (low importance) |
| early_safe_plus_calendar | 14 → 9 | same as early_safe; both calendar features kept |
| full_history | 26 → **10** | crowdfunding, post-IPO (near-constant); log_seed, log_angel, avg_funding_per_round, funding_rounds, n_named_rounds (redundant); several low-importance flags |

**Final early_safe features:** market, country_code, region, n_categories, age_at_first_funding, has_seed, has_angel.
**Final full_history features:** market, country_code, region, n_categories, age_at_first_funding, has_seed, log_funding_total, log_venture, funding_span_years, years_since_last_funding.

Importance (early_safe, AUC drop when shuffled): age_at_first_funding 0.073 > region 0.032 > has_seed 0.016 > has_angel 0.013 > market 0.009 > country 0.008 > n_categories 0.006.
Importance (full_history): log_funding_total 0.037 > age_at_first_funding 0.023 > years_since_last_funding 0.019.

**Preprocessing (`make_preprocessor`):** numeric → median imputation with missing-indicator + standardisation; categorical → constant "missing" + one-hot, with values seen fewer than 50 times grouped as "infrequent". Fitted on train; final matrix widths: early_safe 55, full_history 59.

**Caveat:** selection saw all of the training set, so cross-validation on train is slightly optimistic for the selected features. The held-out test set was not used.
