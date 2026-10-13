"""Readiness-gap classifier: trains on data/labels/readiness_labels.csv (400 YC descriptions, labelled by rubric).
Honest evaluation = 5-fold stratified cross-validation, repeated 3 times, out-of-fold predictions only.
Run: PYTHONPATH=src python -m startupiq.pipelines.readiness_train"""
import json, os, re
import numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

GAPS = ["customer_unclear", "problem_unclear", "traction_not_stated", "differentiation_missing", "business_model_unclear", "team_not_stated"]
RES = "docs/results/readiness"; os.makedirs(RES, exist_ok=True)
df = pd.read_csv("data/labels/readiness_labels.csv")
X_text = df["text"].fillna("").tolist()


def hand(texts):
    f = []
    for t in texts:
        w = t.split()
        f.append([np.log1p(len(w)), bool(re.search(r"\d", t)), bool(re.search(r"[$€£]\s?\d|\d+\s?(m|k|%)\b", t, re.I)),
                  bool(re.search(r"\b(customers?|users?|clients?|revenue|arr|mrr|gmv|paying|growing|growth)\b", t, re.I)),
                  bool(re.search(r"\b(we are|we're|founded|founders?|ex-|former|previously|phd|worked at)\b", t, re.I))])
    return np.array(f, float)


def feats(fit_t, tr_t, te_t):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(tr_t)
    mu, sd = hand(tr_t).mean(0), hand(tr_t).std(0) + 1e-6
    mk = lambda ts: hstack([v.transform(ts), csr_matrix((hand(ts) - mu) / sd)]).tocsr()
    return mk(tr_t), mk(te_t)


rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
res = {}
for g in GAPS:
    y = df[g].values; oof = np.zeros((3, len(y))); strat = y
    for i, (tr, te) in enumerate(rskf.split(X_text, strat)):
        Xtr, Xte = feats(None, [X_text[j] for j in tr], [X_text[j] for j in te])
        m = LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000).fit(Xtr, y[tr])
        oof[i // 5, te] = m.predict_proba(Xte)[:, 1]
    p = oof.mean(0); pred = (p >= 0.5).astype(int); prev = y.mean()
    aucs = [roc_auc_score(y, oof[r]) for r in range(3)]
    base_f1 = f1_score(y, np.ones_like(y))
    res[g] = {"prevalence": round(float(prev), 3), "AUC": round(float(np.mean(aucs)), 3),
              "F1_gap": round(f1_score(y, pred), 3), "F1_all_gap_baseline": round(base_f1, 3),
              "balanced_acc": round(balanced_accuracy_score(y, pred), 3), "accuracy": round(accuracy_score(y, pred), 3),
              "accuracy_majority_baseline": round(max(prev, 1 - prev), 3)}
json.dump(res, open(f"{RES}/cv_results.json", "w"), indent=1)
print(pd.DataFrame(res).T.to_string())
