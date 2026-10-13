# Startup IQ v2: Implementation Plan

Scope: Peer Intelligence MVP, covering syllabus Tasks 4 to 8. Tasks 9 and 10 (integration, deployment, responsible AI) are out of scope here, but every phase below produces inputs for them.

Source: the proposal "Startup IQ: Implementation Proposal for Tasks 4 to 8". Claims about datasets come from that proposal and still need confirmation against the downloaded files.

## 1. Goal and non-goals

**Goal.** Given a company description, return: top-3 sectors with calibrated probabilities (or abstain), nearest peers split into direct and adjacent, a position on a hierarchical market map, a 10th/50th/90th percentile funding range with sample size, and evidence links.

**Non-goals.**

- No startup success, exit, or next-round prediction.
- No news event detection, mentor matching, or portfolio monitoring in this plan.
- No result reported on synthetic data.

## 2. Working rules

1. Raw data is immutable and listed in a manifest (URL, retrieval date, size, checksum, licence note).
2. Every cleaning rule is a scripted step that logs rows before and after.
3. All vectorisers, scalers, and encoders are fitted on training data only.
4. Splits are grouped by company (YC) or issuer (Form D). Two evaluations per stream: stratified random, and temporal.
5. Test sets are scored once.
6. Every model is compared with a stated baseline on the same split, with 95% bootstrap intervals over five seeds.
7. Synthetic data is labelled SYNTHETIC, lives in its own folder, and is excluded from evaluation.
8. Imputation values are computed from training data only and documented per column.

## 3. Proposed repository layout

```
startupiq-v2/
  README.md
  docs/
    IMPLEMENTATION_PLAN.md
    data_cards/            one card per cleaned dataset
    decisions/             short dated decision records
    results/               comparison tables, figures
  data/
    raw/                   immutable downloads plus MANIFEST
    interim/
    processed/
    synthetic/             tests and demos only
  src/startupiq/
    ingest/                download and manifest
    clean/                 one function per cleaning rule
    features/
    models/
    explain/
    evaluate/
    api/                   thin service, added in phase 8
  notebooks/               read-only views of pipeline output
  tests/
  configs/                 YAML per pipeline
```

Tooling is a proposal, not a commitment: Python, scikit-learn, PyTorch for the text CNN and BiLSTM, pandera or Great Expectations for schema checks, a locked environment, and a small web framework for the phase 8 prototype.

## 4. Phases and gates

Each phase ends with a gate. Do not start the next phase until the gate passes.

### Phase 0: Repo and environment

- Create folder and repo, add README and this plan, set up the locked environment and test runner.
- **Gate:** a fresh clone installs and runs an empty test suite with one command.

### Phase 1: Data intake

- Download the YC directory and Form D quarterly files. Write the manifest. Record licence notes.
- Compare real field lists with the proposal. Log every mismatch.
- Record: YC class counts per industry and subindustry, whether Form D flat files encode indefinite offering amounts and how, share of Form D filings that look like operating companies.
- **Gate:** schema matches the proposal or differences are logged. Licence notes recorded. Four open items answered or explicitly parked: YC class balance, Form D indefinite-amount encoding, Form D startup coverage, YC site terms.

### Phase 2: Task 4, EDA and cleaning, Stream A (YC)

- Types, missing values, duplicates (exact then near-duplicate by embedding similarity, reviewed by hand), inconsistencies, outliers on log scale.
- Plots: class counts, tag frequency, description length by industry, companies per batch year, missingness matrix, tag co-occurrence, UMAP of embeddings.
- **Gate:** the cleaned dataset rebuilds from raw with one command. Data-quality log and data card written.

### Phase 3: Task 5, features and splits, Stream A

- Text: cleaned one-liner plus long description, TF-IDF word and character n-grams, pretrained sentence embeddings, token sequences.
- Structured ablation: log team size, stage, number of locations, region.
- Excluded from inputs: tags, subindustry, industries (restate the label), status, isHiring (crawl-time snapshots), founder-demographic flags (audit only).
- Selection: chi-squared, mutual information, L1 or elastic-net, permutation importance. Report whether selection helped.
- Splits: grouped, stratified random plus temporal (older batches train, newest test).
- **Gate:** leakage checklist signed off, including the tag and subindustry exclusion. Pipelines contain all fitted steps.

### Phase 4: Task 6, classical ML, Stream A

- Baselines first: majority class, logistic regression on TF-IDF.
- Candidates: linear SVM on TF-IDF, RBF SVM on embeddings, random forest and gradient boosting, voting or stacking.
- Clustering on embeddings: K-means and agglomerative (Ward). Metrics: silhouette, Davies-Bouldin, bootstrap stability (ARI), NMI and ARI against YC industry, cophenetic correlation.
- Peer retrieval: embedding nearest neighbours versus TF-IDF and BM25, scored by precision@k and nDCG@k with tag overlap as a proxy, plus the hand-rated sample (about 100 queries, top 5 each).
- Error analysis: read about 50 misclassified companies per top confusion pair, sort into label noise, multi-sector, vague text, real error. Accuracy by batch year.
- **Gate:** one comparison table for all models with intervals. Hand-rated sample complete.

### Phase 5: Task 7, deep learning, Stream A

- Primary: text CNN on pretrained word vectors, industry (softmax) and top-50 tags (sigmoid).
- Secondary: BiLSTM with attention, used mainly for the Phase 6 attention test.
- Reference baseline: frozen sentence encoder plus small head.
- Training: AdamW, weight decay, dropout, early stopping on validation macro-F1, class weights, random search, five seeds.
- **Decision rule, fixed before running:** ship the simplest model whose macro-F1 is inside the confidence interval of the best, weighing calibration, latency, and explanation quality. If the network does not earn its place, it stays a documented experiment.
- **Gate:** decision rule applied and recorded in docs/decisions.

### Phase 6: Task 8, interpretability, Stream A

- Linear models: top terms per class.
- Text CNN: filter activations, per-prediction n-gram contributions, gradient saliency as a cross-check.
- BiLSTM: attention heatmaps compared with token-erasure importance.
- Perturbation tests: mask top words, synonyms, typos, one-liner only, remove sector words.
- Reliability: calibration curves and ECE, abstention, accuracy by batch year, region, and description length, an out-of-domain text test, subgroup error rates using the excluded founder flags.
- Hand-rate 30 explanations.
- **Gate:** comparison table with predictive, interpretability, robustness, and cost columns for every model.

### Phase 7: Stream B (Form D), Tasks 4 to 6 and 8

- Cleaning: join by accession number, keep latest amendment, drop funds and non-operating issuers, check sentinels, log-scale amounts, consistency checks.
- Features: industry group, company age at filing, revenue range (ordinal, "Decline to Disclose" separate), entity type, state or country, filing year and quarter, exemption and security type, related-person count. Excluded: amount sold when offering amount is the target and the reverse.
- Splits grouped by issuer; temporal holdout.
- Models: segment-median baseline, linear regression and ridge, decision tree on quartile bands, random forest, gradient boosting, quantile models for the 10th, 50th, 90th percentiles.
- Metrics: MAE and RMSE on log scale, R-squared, median absolute percentage error, macro-F1 for bands, quantile coverage.
- Explanation: permutation importance, SHAP, partial dependence, tree rules, coefficients with intervals, residuals by industry, age, and amount range.
- **Gate:** beats the segment-median baseline, or ships as empirical segment quantiles.

### Phase 8: Crosswalk and prototype

- Build the sector crosswalk (YC sectors to Form D industry groups) by hand, review it, and document its coarseness.
- Thin service: description in; sector probabilities (or abstain), direct and adjacent peers, market-map position, funding range with sample size, evidence links out.
- **Gate:** end-to-end run on representative inputs returns all outputs with evidence. This is the input to Task 9.

## 5. Definition of done per phase

A phase is done when its code, tests, logs, data card or results table, and a dated decision record are committed, and the gate has been checked by someone other than the author of the step.

## 6. Missing-data policy (summary)

| Case | Rule |
| --- | --- |
| Empty long description | Fall back to one-liner, flag the row |
| Team size zero | Treat as missing; impute from training median or keep a missing marker |
| "Decline to Disclose" revenue range | Keep as its own category |
| Mostly empty column | Drop, log the reason |
| Rare sector with too few examples | Merge into a larger class |
| Synthetic rows | Training only if justified and reported; never in test sets |

## 7. Risks and open decisions

| Item | Status | Owner decision needed |
| --- | --- | --- |
| YC data has no stated licence | Open | Experiments only until permission or replacement text source |
| Form D licence not stated on SEC page | Open | Legal read before launch |
| Whether a pretrained transformer may count for Task 7 | Open | Syllabus check; plan assumes text CNN is the graded model |
| Source of an incubator cohort's own funding amounts | Open | Not specified in the proposal |
| YC is US-centred, one accelerator | Known limit | Out-of-domain test; widen corpus before launch |
| Small data for deep learning | Known limit | Decision rule fixed in advance |
| Peer quality has no ground truth | Known limit | Hand-rated sample; state the proxy |

## 8. Deferred

Success or exit prediction, news event detection (needs about 1,000 to 2,000 labelled headlines), mentor and resource matching, portfolio monitoring, trend forecasting.
