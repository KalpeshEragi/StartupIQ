# Task 5: Features, feature selection, and data splits

Images are in this folder. Code: `src/startupiq/pipelines/task56_yc.py` (YC industry features and selection), `readiness_export.py` (readiness features), `task56_formd.py` (Form D features and grouped split), `output_figures.py` (draws the images). Numbers: `docs/results/task5_6/results_yc.json`, `results_formd.json`, `docs/results/readiness/hybrid_results.json`.

## What Task 5 is, in one line
Decide what the models are allowed to see (features), keep only the useful ones (selection), and split the data so the final score is honest.

---

## Image 1: `1_inputs_used_and_leakage_exclusions.png` (input features: used vs excluded)
**What it shows.** A table of inputs and the decision for each. USED: the description text, plus length, has-a-number and has-a-money-figure signals. TESTED: team size and stage, which lowered the score (validation macro-F1 0.724 against 0.741 without them). EXCLUDED: YC tags, subindustry/industries, status and isHiring, and (for Form D) amount sold. AUDIT ONLY: founder demographic flags.
**Why.** "Leakage" means a feature secretly contains the answer. Five YC tags are literally industry names, and subindustry restates the industry, so using them would make the industry model look great and be useless on a real founder's text. Status and isHiring were captured at crawl time, not when the company was founded. Amount sold is the same event as the amount we predict for Form D.
**What to say.** "We could have reached higher accuracy by using tags, but that would be cheating. I excluded them on purpose."
**Likely question.** *Why keep demographic flags out?* They are not inputs. We keep them only to check later whether the model treats groups differently.

## Image 2: `2_feature_sets_and_selection.png` (feature sets and feature selection)
**What it shows.** Left: validation macro-F1 for four feature sets in the YC industry model. Word features alone win (0.741); adding character patterns (0.729), team size and stage (0.724), or both (0.702) made things worse. Right: chi-squared selection keeping the top k features. The best is 5,000 of 24,569 features (0.760); keeping all is 0.741; keeping only 500 drops to 0.664.
**Also tried.** L1 (lasso-style) selection kept 3,829 features and scored 0.657, so it was not used.
**Why.** More features is not automatically better; small data plus many features invites overfitting. We only keep a feature set that wins on validation data.
**What to say.** "Selection gave about two points. It is a small gain, and I report it as small."

## Image 3: `3_readiness_labels_and_features.png` (readiness labels and length feature)
**What it shows.** Left: how common each gap is among the 400 labelled descriptions: who-it's-for 9%, problem 19%, why-you 16%, traction 82%, how-you-earn 91%, team 94%. Middle: the share of texts with each of the first three gaps, by description-length quartile; shorter texts have more gaps. Right: cross-validated AUC with and without the length and number features: 0.82 to 0.83 (customer), 0.76 to 0.83 (problem), 0.69 to 0.77 (differentiation).
**Why.** Traction, business model and team are "missing" in 82 to 94% of texts. A model cannot learn from a label that is almost always true, so those three use transparent keyword rules. The other three use a trained model. Length matters because short descriptions simply say less; without length features the model almost never flagged a gap (recall of 6%, 12% and 2%; with length: 51%, 64% and 53%).
**What to say.** "I defined the labels myself against a six-point rubric, found the three skewed ones are better as rules, and showed with numbers that length carries real signal."
**Honest limit.** The labels are one person's judgement (mine), so scores measure agreement with that rubric, not startup outcomes. I read the 400 texts while writing the rules, so rule scores are somewhat optimistic.

## Image 4: `4_train_validation_test_splits.png` (train / validation / test splits)
**What it shows.** YC industry model: stratified random split 4,349 / 933 / 933 (train / validation / test), plus a temporal split (4,846 older companies train, 1,369 from 2025 onward test). Form D: split grouped by issuer, 6,611 / 1,657 / 2,053, so no company is in two sets. Readiness check: 5-fold cross-validation repeated three times, with every text scored by models that never saw it.
**Why.** The test set was scored once. Scalers, vectorisers and selectors were fitted on training data only, because fitting them on all the data leaks test information into training. Temporal and grouped splits stop a model from "remembering" the same company or the same era.
**What to say.** "Every accuracy number in Task 6 comes from data the model never trained on."

---

## Things not done in Task 5 (say this if asked)
No deep-learning features (Task 7), and no explanation of individual predictions yet (Task 8).
