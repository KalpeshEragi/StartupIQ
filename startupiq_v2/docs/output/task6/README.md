# Task 6: Classical machine learning models

Images are in this folder. Code: `src/startupiq/pipelines/readiness_export.py` (readiness model), `task56_yc.py` (industry models), `task56_formd.py` (funding models), `output_figures.py` (draws the images). Numbers: `docs/results/readiness/hybrid_results.json`, `docs/results/task5_6/results_yc.json` and `results_formd.json`.

## What Task 6 is, in one line
Train several classical models, compare them to simple baselines on data they never saw, and report the honest result.

## How this connects to the product
The readiness model (image 1) is the one the web page uses. The industry model (images 2 and 3) is a supporting experiment: it routes a description to a market group. The funding models (image 4) feed the "what do companies like me raise" benchmark.

---

## Image 1: `1_readiness_model_results.png` (readiness model results)
**What it shows.** Left: cross-validated AUC for the three trained checks: who-it's-for 0.83, problem 0.83, why-you 0.77 (0.50 is chance). Right: balanced accuracy for all six checks: 0.73, 0.75 and 0.71 for the trained model; 0.72 (traction), 0.76 (how you earn) and 0.74 (team) for the rule checks.
**How to read it.** AUC is how well the model ranks texts that have a gap above texts that do not. Balanced accuracy averages the hit rate on gap texts and on no-gap texts, so it is not fooled by rare or very common gaps.
**Why not plain accuracy.** Plain accuracy for the trained checks is 0.90, 0.82 and 0.84, about the same as always guessing the majority (0.91, 0.82, 0.84). So I do not use it as the headline. For the rule checks, plain accuracy is below the majority guess (0.74, 0.80 and 0.89 against 0.82, 0.91 and 0.94), but they detect gaps with high precision (92 to 97%) and recall of 75%, 81% and 91%.
**What to say.** "The model finds the three softer gaps well above chance. The three skewed gaps are handled by rules because a model cannot beat the majority guess there."
**Honest limits.** One labeller, 400 texts, and I wrote the rules after reading them. The scores are an estimate of agreement with the rubric.

## Image 2: `2_industry_model_comparison.png` (industry model results)
**What it shows.** Left: six models on the validation set. Always guessing B2B: accuracy 0.51, macro-F1 0.09. Logistic regression 0.79 / 0.76, linear SVM 0.80 / 0.75, complement naive Bayes 0.78 / 0.70, random forest 0.73 / 0.54, soft-voting of logistic regression and naive Bayes 0.80 / 0.77 (accuracy / macro-F1). The voting model won on validation. Right: the chosen model on the test set, scored once: accuracy 0.80 (95% interval 0.78 to 0.82), macro-F1 0.70 (0.62 to 0.75), top-3 accuracy 0.98.
**Extra results.** On the temporal test (newest 2025+ companies) accuracy is 0.83 and macro-F1 0.74. When the model only answers if its confidence is 0.6 or higher, it answers 82% of the time and is right 86% of those times.
**What to say.** "Accuracy reached 80%, but the interval goes down to 78%, and macro-F1 is lower (0.70) because small industries are harder. The right answer is in the model's top 3 guesses 98% of the time."
**Why this model matters less to the product.** A founder already knows their industry, so this is used to route to peer groups, not to tell founders what they do.
**Not shown.** Clustering the descriptions found weak structure (silhouette around 0.03), so I do not claim clean market groups.

## Image 3: `3_industry_confusion_matrix.png` (test confusion matrix)
**What it shows.** For the chosen industry model on the test set, rows are the actual industry and columns are the predicted one. The diagonal is correct predictions; each off-diagonal cell is a specific mix-up. Colour is the share of each row.
**How to use it.** Point at the big B2B row (the largest class), then at the small industries (education, real estate, the merged "Other") where the diagonal is weaker. That visually explains why macro-F1 is below accuracy.
**Note.** This image is the plain output of the pipeline run. I have not written a cell-by-cell summary of it, so read the actual confusions off the picture before presenting.

## Image 4: `4_formd_funding_models.png` (funding model results)
**What it shows.** Left: accuracy for predicting a low / mid / high funding band (band edges $1.9M and $6.7M, from training data) on validation: majority guess 0.35, logistic regression 0.52, decision tree 0.52, random forest 0.55, histogram gradient boosting 0.55. Middle: on the test set, the chosen model gets 0.54 against 0.35 for guessing, and 0.90 are within one band. Right: for predicting the amount itself (log scale), R-squared is 0.25 for gradient boosting against 0.04 for the segment-median baseline (MAE on log scale 0.55 against 0.64).
**Quantile range.** A 10th to 90th percentile range from quantile models covers 76% of true amounts against an 80% target, a median width of about 1.5 on the log10 scale (roughly 30 times from low end to high end).
**What to say.** "Form D filings only tell us industry group, age, revenue range and similar facts. That explains only a quarter of the variation in raise size. The model beats the baseline, but modestly. For founders, the right product is the range that comparable companies raised, with the sample size, not a single predicted number."
**Honest limit.** This is below the 80% target. I did not push it up with leaky features such as amount sold, which would inflate it without meaning anything.

---

## Summary table to show the mentor
| Model | Data | Headline result | Baseline |
| --- | --- | --- | --- |
| Readiness check (3 trained gaps) | 400 labelled YC texts, cross-validated | AUC 0.77 to 0.83, balanced accuracy 0.71 to 0.75 | chance 0.50 |
| Readiness check (3 rule gaps) | same | balanced accuracy 0.72 to 0.76 | 0.50 |
| Industry classifier | 6,215 YC texts, test set | accuracy 0.80, macro-F1 0.70, top-3 0.98 | 0.51 accuracy, 0.09 macro-F1 |
| Funding band | 10,321 Form D filings, test set | accuracy 0.54 (0.90 within one band) | 0.35 |
| Funding amount | same | R-squared 0.25 | 0.04 |
