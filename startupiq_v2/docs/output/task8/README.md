# Task 8: Interpretability and reliability (YC industry models)

Question: can we trust and explain the models? Computed on test texts with the Task 7 models.

## Images
1. `1_perturbation.png`: macro-F1 on 300 test texts after changes. Typos change little (LR 0.69 to 0.70). Removing sector words drops LR to 0.40, CNN 0.32, BiLSTM 0.26; masking the top-3 words drops LR to 0.45. Say: "models depend on obvious domain words, which is expected, but they are fragile without them."
2. `2_calibration.png`: confidence vs accuracy. ECE (lower is better): LR 0.062, BiLSTM 0.072, CNN 0.098. Say: "LR's confidence is the most honest."
3. `3_accuracy_by_length.png`: LR short 0.75, medium 0.83, long 0.82. Short descriptions are the weak spot for every model.
4. `4_explanation_agreement.png`: agreement between highlighted words and token erasure (Spearman): BiLSTM attention 0.07, CNN saliency 0.13. Say: "attention heatmaps are not a reliable explanation here, so we explain with LR terms."

## Other results (in the JSON)
- LR top terms per industry look sensible (Education: students, school, teachers; Fintech: payments, banking, insurance).
- Abstaining below 0.6 confidence: LR keeps 73% of texts at 87.7% accuracy.
- Out-of-domain texts (garage, bakery, poem, yoga): LR stayed under 0.6 confidence on all four; the CNN gave 0.83 on one.
- Accuracy before 2025 vs 2025 on: LR 0.793 / 0.816.

## Likely questions
- Why trust LR? It is the best scorer, best calibrated, and its terms can be read directly.
- Are the explanations verified by people? Not yet: `docs/results/task8/explanations_to_rate.csv` has 30 examples waiting for a human rating.

## Limits
Out-of-domain test is only 4 sentences. No subgroup analysis by region or founder flags yet. Perturbations use a 300-text sample.
