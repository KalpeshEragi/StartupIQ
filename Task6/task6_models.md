# Task 6 — Models

Tuning: `GridSearchCV`, 5-fold stratified, scored by ROC-AUC (threshold-free). Classification threshold fixed at 0.5. Full table: `Task6/outputs/model_results.csv`.

| Model | Purpose | Grid searched | Chosen (early_safe) |
|---|---|---|---|
| Majority class | reference | – | acc 0.586, macro-F1 0.370, AUC 0.5 |
| Logistic regression | interpretable baseline | C ∈ {0.1, 1} | C = 0.1 |
| Random forest (300 trees) | interactions / non-linearity | min_samples_leaf ∈ {5, 20}, max_depth ∈ {8, None} | leaf 5, depth None |
| HistGradientBoosting | boosted trees | max_depth ∈ {3, None}, min_samples_leaf ∈ {20, 50}, lr 0.05 | depth None, leaf 50 |

## ROC-AUC by feature set (train / 5-fold CV / test), best-by-CV model per set
| Feature set | Model | Train | CV | Test |
|---|---|---|---|---|
| profile_only | random forest | 0.738 | 0.708 | 0.717 |
| **early_safe (primary)** | HistGradientBoosting | 0.789 | 0.724 | 0.734 |
| early_safe + calendar | random forest | 0.848 | 0.754 | 0.769 |
| full_history (leaky) | HistGradientBoosting | 0.828 | 0.803 | 0.797 |

Model choice used CV only. On early_safe, RF (CV 0.7225, test 0.741) and HGB (CV 0.7241, test 0.734) are indistinguishable within CV noise (std ≈ 0.01).

## Overfitting
- Random forests overfit: early_safe RF train AUC 0.81 vs CV 0.72; full-history RF 0.90 vs 0.80. Their tuned settings sit at the edge of the small grid (unrestricted depth, leaf 5), so a bigger grid might trim the gap slightly but not change the ranking.
- Logistic regression has a small train-CV gap (0.72 vs 0.71) but lower scores.
- CV ≈ test for every model (differences ≤ 0.02), so the test result is consistent with validation. CV is mildly optimistic because feature selection saw the training set (Task 5).
