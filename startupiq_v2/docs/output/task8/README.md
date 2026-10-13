# Task 8: Interpretability and reliability, for the Competitor Finder

## Why this task exists in StartupIQ (plain words)
- The founder sees a list of competitors. Two questions decide whether they can trust it:
  1. **Why is this company on my list?** (explanation)
  2. **When should I NOT trust the list?** (reliability)
- Task 8 answers both for the shipped finder (LSA), and also explains the CNN to compare.

## Where the code is
| What | File |
|---|---|
| All checks | `src/startupiq/pipelines/task8_competitors.py` |
| Images | `src/startupiq/pipelines/output_figures_comp.py` |
| Numbers | `docs/results/task8_competitors/results_task8_competitors.json` |
| Readable examples (10 startups x 5 competitors, with "matched on" words) | `docs/results/task8_competitors/worked_examples.md` |
| 150 pairs for hand rating | `docs/results/task8_competitors/explanations_to_rate.csv` |

## Example of what the founder sees
> **MySivi** (AI English speaking app) -> Toko (shared tag: AI-Enhanced Learning; matched on: app, English), Edwin (matched on: learning, Tutor, AI-powered), Speak (matched on: AI-powered, Tutor, AI)

## Images
1. `1_explanation_faithfulness.png`: delete the 3 "matched on" words from the query and similarity to the top competitor drops 0.265 (LSA); deleting 3 random words drops only 0.013, about 20 times less. Say: "the words we show really are the reason."
2. `2_robustness.png`: top-10 list that stays after changing the description: typos 83%, 30% of words removed 62%, first sentence only 32%. Say: "short, one-line descriptions give a very different list, so ask founders for a full description."
3. `3_confidence_vs_precision.png`: precision@10 rises from 0.30 (lowest-similarity fifth) to 0.49 (highest). Similarity is a usable confidence signal.
4. `4_quality_by_industry.png`: precision@10 is highest for Real Estate (0.51), Industrials (0.48), B2B (0.44) and lowest for Fintech (0.32) and Education (0.32).

## Code snippets (copied from the real files)

**1. "Why" words = delete one word at a time, see how much the match falls**
```python
def occlusion(enc_name, text, nb_vec):
    w = text.split()[:100]; base = float(ENC[enc_name]([" ".join(w)])[0] @ nb_vec)
    alts = [" ".join(w[:i] + w[i + 1:]) for i in range(len(w))]
    return w, base, base - (ENC[enc_name](alts) @ nb_vec)     # big drop = matched on this word
```

**2. Faithfulness test: top-3 "why" words vs 3 random words**
```python
top3 = set(np.argsort(-drop)[:3]); rnd = [set(rng.choice(len(w), 3, replace=False)) for _ in range(3)]
dr_top.append(base - sim(top3)); dr_rand.append(np.mean([base - sim(r) for r in rnd]))
```

**3. Robustness: how much of the list survives a perturbation**
```python
ov = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(base_top, top2)])
```

**4. Abstain when the best match is weak (threshold from VALIDATION only)**
```python
thr = float(np.quantile(Sv.max(1), 0.2))        # 20th percentile of best-match similarity on val
keep = top1 >= thr                              # test: coverage 0.855, precision@10 0.423 kept vs 0.308 abstained
```

## Other results
- By description length precision@10: short 0.37, medium 0.43, long 0.42. By era: before 2025 0.39, 2025 on 0.47.
- Abstaining on the weakest 20% of cases lifts precision@10 on the kept ones to 0.42.

## Honest weak spot
- **The finder does not know when text is off-topic.** Four non-startup sentences (garage, bakery, poem, yoga) got best-match similarity 0.58, 0.58, 0.85 and 0.50 against a normal median of 0.57, and none fell below the abstain threshold. A separate "is this a startup description?" check is needed before launch.

## Likely questions
- **Is the explanation causal?** It is a deletion test on the similarity score itself, so it explains the finder, not the real world.
- **Why not attention or gradients?** The earlier industry-classifier experiment showed attention agreed with deletion only 0.07 (archived in `docs/output/_archive_industry_classifier/`).
- **Has a human checked the explanations?** Not yet. `explanations_to_rate.csv` is waiting for ratings.

## Limits
Relevance is judged by shared tags only; the off-topic test uses 4 sentences; robustness uses 300 queries; no subgroup analysis by region or founder flags yet.
