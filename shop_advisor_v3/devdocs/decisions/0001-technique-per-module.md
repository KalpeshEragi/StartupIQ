# 0001: Which technique each module uses

- Status: Accepted (design); individual model choices have their own records
- Date: 2026-10-24
- Module / task: all; Task 9 architecture

## Problem

The syllabus asks for rules (Module 1), classical ML (Module 2), deep learning (Module 3) and an integration of all three (Module 4). Using a technique only because the syllabus names it would make a worse product and a weaker viva answer. Each module needs the technique that fits its problem.

## Principle

Use the simplest technique that is correct, and only add learning where there is a pattern to learn and data to learn it from.

## Choices

| Module | Technique | Why this | Why not the alternative |
| --- | --- | --- | --- |
| M0 intake | Deterministic rules | Column roles and data tiers must be predictable and explainable to the user | A learned column classifier needs labelled messy files we do not have; a wrong silent guess is worse than asking |
| M1 vitals | Formulas | Runway = cash / burn is exact | Any model would add error to an exact quantity |
| M2 anomalies | Isolation Forest (ML) | No labels for "anomaly"; works on small series; fast | Autoencoder (DL) needs more data per series; kept as stretch comparison |
| M3 forecast | LSTM and 1D-CNN (DL) against LightGBM and seasonal baselines | Sequences are the data type the syllabus pairs with RNN/LSTM; a global model can learn shared patterns across 1,115 stores | See 0005 for the full comparison and the pre-registered decision rule |
| M4 reviews | Text CNN (DL, stretch) | Short texts, local phrases ("entrega atrasada") carry the signal | Transformer is heavier and the syllabus asks for CNN/RNN |
| M5 market pulse | Decomposition + classifier (ML) | Regimes (rising, flat, falling, seasonal) are a small tabular problem | DL is overkill for a handful of features |
| M6 peer cohort | K-means and Ward clustering (ML) | No labels; segments must be explainable | — |
| M7 rival radar | Gradient boosting / random forest / logistic (ML) | Tabular features, about 13k labelled rows | DL on 13k tabular rows rarely beats boosting; see 0006 |
| M8 diagnosis | Rule-based expert system | Advice must be traceable to conditions; founders and examiners can read the rules | A learned policy has no ground truth to learn from and cannot explain itself |
| M9 planner | State-space search | Choosing a sequence of moves under a budget is a planning problem | Reinforcement learning needs a trustworthy simulator and much more data |
| M10 safety | Rules | Hard constraints (no price coordination, abstain on thin data) must never be "probably" enforced | — |
| M11 note | Templates | Text must say exactly what the rules found | A language model could invent reasons not in the trace |

## How the parts connect

Learned modules output `Fact`s with a certainty. Rules consume facts, so a model can be replaced without rewriting rules. Search uses the forecaster as its transition model. This is the "DL for representation, ML for prediction, AI for decisions" structure Task 9 asks for.
