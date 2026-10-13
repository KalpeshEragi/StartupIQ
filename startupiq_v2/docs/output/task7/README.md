# Task 7: Deep learning (YC industry classification)

Question: does a neural network beat the simple TF-IDF + logistic regression (LR) on the same split?
Models: text CNN, BiLSTM with attention (embeddings learned from scratch; no pretrained vectors offline). Same split as Task 5; test scored once.

## Images
1. `1_test_macro_f1.png`: LR 0.689, CNN 0.649, BiLSTM 0.552 (bars = 95% bootstrap CI). Say: "macro-F1 averages all industries equally, so it is the fair score."
2. `2_test_accuracy.png`: LR 0.798, CNN 0.741, BiLSTM 0.650. Say: "LR is best on both scores; the CNN's interval is lower."
3. `3_cnn_random_search.png`: 4 random CNN settings (validation macro-F1 0.59 to 0.70) vs LR's 0.741 dashed. Say: "tuning helped the CNN but none reached LR."
4. `4_cost.png`: LR 0.2M parameters / ~5 s; CNN 1.3M / ~92 s; BiLSTM 1.3M / ~49 s.

## Decision (rule fixed before running)
Ship the simplest model whose validation macro-F1 is at or above the lower 95% bound of the best. LR was best on validation (0.741, bound 0.671), so LR ships.

## Likely questions
- Why did deep learning lose? About 4,300 training texts is small; the labels are driven by clear keywords, which TF-IDF already uses.
- Did you tune enough? Limited: 4 CNN trials, 3 CNN seeds, 2 BiLSTM seeds (time limit of the machine). A bigger search or pretrained vectors could close the gap; not tested.
- Is the test set clean? Yes, used once after the decision.

## Limits
Embeddings from scratch; reduced search/seeds; one split. Results: `docs/results/task7/results_task7.json`.
