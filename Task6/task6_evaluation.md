# Task 6 — Evaluation and error analysis (test set, 1,259 companies)

Positive class = acquired. Threshold 0.5.

| Model / set | Acc | Prec | Recall | F1 | Macro-F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Majority class | 0.586 | 0.586 | 1.000 | 0.739 | 0.370 | 0.500 | 0.586 |
| **early_safe, HGB (primary)** | 0.666 | 0.752 | 0.644 | 0.693 | 0.664 | 0.734 | 0.769 |
| early_safe, RF | 0.678 | 0.747 | 0.680 | 0.712 | 0.673 | 0.741 | 0.778 |
| early_safe, LR | 0.678 | 0.753 | 0.671 | 0.710 | 0.674 | 0.732 | 0.773 |
| full_history, HGB (leaky) | 0.723 | 0.785 | 0.726 | 0.754 | 0.718 | 0.797 | 0.835 |
| full_history, RF (leaky) | 0.743 | 0.792 | 0.762 | 0.776 | 0.737 | 0.800 | 0.834 |

Note: the majority classifier has a higher positive-class F1 (0.739) than the early model, which is why macro-F1 and ROC/PR-AUC are the metrics to read here.
Confusion matrix, primary model: TN 364, FP 157, FN 263, TP 475 (`figures/confusion_matrices.png`; ROC/PR in `figures/roc_pr_curves.png`).

## What the feature-set comparison shows
- Prediction-safe features give ROC-AUC ≈ 0.73: real but modest signal, above profile-only (0.72) and majority (0.50).
- Full history adds ~0.06 AUC (0.80). This is an upper bound that includes future information; it is not usable at evaluation time.
- Calendar year adds ~0.03 (0.77), a cohort effect. Time-based check on the primary features (train on founded ≤ 2007, test on ≥ 2008): AUC 0.711, macro-F1 0.615, recall for acquired only 0.41, because the acquired share drops from 71% to 45% across periods.

## Error analysis (primary model): 263 false negatives (acquired predicted closed), 157 false positives
- **Missing dates:** where `age_at_first_funding` is missing (24% of test), error is 41% vs 31% otherwise, mostly false negatives (33%).
- **Founding era:** best on ≤2000 cohorts (21% error, 86% acquired); worst where founding year is missing (42%). The model leans on cohort-correlated signals.
- **Age at first funding:** false positives are highest for companies that raised 2–5 years after founding (21% FP rate, 70% actually acquired); false negatives concentrate in ≤2 years (21%).
- **Market:** Enterprise Software is easy (12% error, mostly acquired); Curated Web (47%) and Biotechnology (46%) are near coin-flips. Software false negatives 23%.
- **Country:** US errors split between false positives (15%) and false negatives (18%); missing country is mostly false negatives (33%); UK error is low (20%).
- **Typical false positives** are older companies with long lead times that closed (e.g. PlumTV, Etelos, OpVista); **typical false negatives** are young seed-backed companies acquired quickly (Zopim, Purewire, Semantria).
- **Early vs full:** the full model fixes 197 of the early model's 420 mistakes but introduces 126 new ones; 223 are wrong in both, so about 18% of companies are hard for either.
- **Test-set importance (early_safe):** age at first funding (−0.097 AUC when shuffled), region (0.039), country (0.029), seed flag (0.026), market (0.024).

## Limitations
- Modest discrimination (AUC ~0.73); not sufficient alone for an investment decision.
- The population is funded companies with a resolved outcome, up to 2014; acquired share depends strongly on cohort, and current startups are outside the observed period.
- `operating` companies (85% of the data) are excluded from training; the model says nothing about their future beyond the closed/acquired contrast.
- No team, traction, revenue, investor-quality or text features exist in this dataset; the tags are as of the snapshot.
- Seed/angel amounts are cumulative across rounds; the early-safe set is a reasonable but imperfect approximation of "known at first funding".
