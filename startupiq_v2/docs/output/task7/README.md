# Task 7: Deep learning, for the Competitor Finder

## Why this task exists in StartupIQ (plain words)
- Product promise: a founder types what their startup does, and we show **their competitors**.
- Task 7's job: test whether a **neural network (text CNN)** finds better competitors than simpler methods.
- How the CNN is trained: it reads a startup description and predicts its **tags** (e.g. Payments, DevOps, Marketplace). It must learn what each description is "about". Its internal vector (384 numbers) is then used as the startup's "fingerprint"; the closest fingerprints are the competitors.
- Compared against: **TF-IDF** (word overlap), **LSA** (TF-IDF squeezed to 200 topic numbers), and a **Hybrid** (CNN + TF-IDF).
- Competitor = startup that shares at least one specific tag with the query (industry-name tags such as Fintech/B2B are removed so it is not trivial).

## Where the code is
| What | File |
|---|---|
| CNN, multi-label training, embeddings | `src/startupiq/models/textnn.py` |
| Data, tag relevance, precision@10 | `src/startupiq/pipelines/comp_common.py` |
| Experiment + decision rule | `src/startupiq/pipelines/task7_competitors.py` |
| Images | `src/startupiq/pipelines/output_figures_comp.py` |
| Numbers | `docs/results/task7_competitors/results_task7_competitors.json` |

## Images
1. `1_competitor_precision.png`: test precision@10. Random 0.13, TF-IDF 0.38, **LSA 0.41**, CNN 0.35, Hybrid 0.42 (bars = 95% CI, 778 test startups). Say: "every method is about 3x better than random; the CNN is the weakest."
2. `2_tag_prediction.png`: how well each model predicts the 50 tags. CNN micro-AP 0.26 vs TF-IDF logistic regression 0.34 (random 0.03); AUC 0.81 vs 0.89.
3. `3_decision_validation.png`: the rule was fixed before running: ship the simplest method whose validation precision@10 is at or above the lower 95% bound of the best (0.405). LSA is best (0.423) and TF-IDF (0.394) falls below the line, so **LSA ships**. The Hybrid (0.416) also passes, but is more complex.
4. `4_same_industry.png`: sanity check, share of the 10 competitors in the same industry: LSA 0.67, TF-IDF 0.63, Hybrid 0.65, CNN 0.56.

## Code snippets (copied from the real files)

**1. Tags the CNN must predict (industry-name tags removed, so no free hints)**
```python
industry = set(df["industry"].unique()) | {"B2B", "Consumer"}
d["taglist"] = [[t for t in parse_tags(s) if t not in industry] for s in d["tags"]]
top = [t for t, _ in Counter(t for tl in tr["taglist"] for t in tl).most_common(50)]
```

**2. Train the CNN with a sigmoid per tag (a startup can have several tags)**
```python
lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)          # rare tags weigh more
f = average_precision_score(Yva.numpy(), predict_sigmoid(model, Xva), average="micro")
if f > best: best = f; state = copy_of_weights           # early stopping on validation
```

**3. The fingerprint = the CNN's pooled features, centred and length-normalised**
```python
def embed(model, X):   # textnn.py
    return model.pooled(model.emb(X))                    # 384 numbers per startup
E = embed(model, X); vectors = normalize(E - E_train_mean)
```

**4. Find competitors = nearest fingerprints (cosine), ignoring duplicates of the same company**
```python
S = query_vectors @ corpus_vectors.T                     # similarity of every pair
S[dup] = -1e9                                            # same domain / near-duplicate text excluded
top10 = np.argsort(-S, axis=1)[:, :10]
```

**5. How good is the list: precision@10**
```python
rel = (T_query @ T_corpus.T) > 0                         # share at least one tag
p10 = rel[rows, top10].mean(1)                           # share of the 10 that are relevant
```

**6. The decision rule (validation only; test touched once afterwards)**
```python
best = max(names, key=lambda n: val[n]["p10"].mean()); lo = boot_ci(val[best]["p10"])[0]
chosen = next(n for n in ["TFIDF", "LSA", "CNN", "HYBRID"] if val[n]["p10"].mean() >= lo)
```

## Likely questions
- **Why not just use the CNN?** It did not earn its place: lower precision than LSA (paired difference to TF-IDF -0.046 to -0.011) with about 1.3M parameters. About 2,000 tagged startups is too little for a network.
- **Is "shares a tag" a fair test?** It is conservative. In the examples some correct competitors share no tag (e.g. Speak for an English-speaking app), so true quality is likely higher than 0.41. It is a proxy, not ground truth.
- **Why remove industry tags?** Otherwise "same industry" would make the task trivial.

## Limits
Tags are founder-chosen and inconsistent; only YC startups are in the index (US-centred); CNN embeddings are learned from scratch (no pretrained vectors offline); 2 CNN seeds and one configuration only. The earlier industry-classifier experiment is kept in `docs/output/_archive_industry_classifier/`.
