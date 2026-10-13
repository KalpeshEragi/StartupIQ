# Task 8: Interpretability and reliability (YC industry models)

## What we did, in plain words
- Task 7 gave us three models. Task 8 asks: **can we explain their answers, and can we trust them?**
- Four checks:
  1. **Explain**: which words push a prediction? (LR top terms, CNN saliency, BiLSTM attention)
  2. **Do the explanations hold up?** Compare "highlighted words" with a simple test: delete each word and see how much confidence drops (token erasure).
  3. **Break it on purpose**: typos, remove industry words, keep only the first sentence, mask the top 3 words.
  4. **Reliability**: is confidence honest (calibration), is it worse for short texts, and does it admit uncertainty on off-topic text?
- Result: LR is easiest to explain, best calibrated and most accurate. Attention heatmaps barely agree with erasure (0.07), so we do not use them as explanations.

## Where the code is
| What | File |
|---|---|
| All checks | `src/startupiq/pipelines/task8_yc.py` |
| Images | `src/startupiq/pipelines/output_figures_78.py` |
| Numbers | `docs/results/task8/results_task8.json` |
| 30 explanations to rate by hand | `docs/results/task8/explanations_to_rate.csv` |

## Code snippets (copied from the real files)

**1. LR top terms per industry**
```python
fn = np.array(tf.get_feature_names_out())
out["lr_top_terms"] = {c: fn[np.argsort(-lr.coef_[i])[:8]].tolist() for i, c in enumerate(classes)}
```

**2. Token erasure: delete one word at a time, see how much the predicted class drops**
```python
def erasure(model_key, text):
    toks = tokenize(text)[:100]; base = P[model_key]([text])[0]; k = int(base.argmax())
    alts = [" ".join(toks[:i] + toks[i + 1:]) for i in range(len(toks))]
    return base[k] - P[model_key](alts)[:, k]          # big drop = important word
```

**3. CNN saliency (gradient x input) and the agreement test**
```python
e = cnn.emb(x).detach().requires_grad_(True)
lg = cnn(x, e=e); k = int(lg.argmax()); lg[0, k].backward()
s = (e.grad * e).sum(-1).abs()[0]                      # importance per word
r = spearmanr(a[0, :n].numpy(), erasure_scores).statistic   # attention vs erasure
```

**4. Perturbation test (image 1)**
```python
def drop_sector(t): return " ".join(w for w in t.split() if clean(w) not in sector)
row[name] = round(mf1(sy, P[m](mk(m)).argmax(1)), 3)   # macro-F1 after the change
```

**5. Calibration error (image 2)**
```python
def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1); e = 0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e                                            # 0 = confidence matches accuracy
```

**6. Abstain when unsure**
```python
keep = probs[m].max(1) >= 0.6      # LR: keeps 73% of texts at 87.7% accuracy
```

---

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
