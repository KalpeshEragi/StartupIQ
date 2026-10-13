"""Hybrid readiness model: ML (TF-IDF + logistic regression) for customer / problem / differentiation,
transparent rules for traction / business model / team. Evaluates both honestly, exports model JSON for the web page.
Run: PYTHONPATH=src python -m startupiq.pipelines.readiness_export"""
import json, re, os
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold

RES = "docs/results/readiness"; os.makedirs(RES, exist_ok=True)
df = pd.read_csv("data/labels/readiness_labels.csv"); T = df["text"].fillna("").tolist()

RULES = {
 "traction_not_stated": r"(\d\s?%|[$€£]\s?\d|\b\d+(\.\d+)?\s?(k|m|b|million|billion)\b|\b\d{2,}[,.]?\d*\+?\s+(users|customers|companies|clients|teams|downloads|stars)\b|\b(paying customers?|revenue|arr|mrr|gmv|acquired|backed by|raised|series [a-c]|fortune 500|waitlist|pilots?|live with|launched|growing)\b)",
 "business_model_unclear": r"\b(subscription|per month|monthly|pricing|price|fees?|commission|saas|marketplace|we sell|sells?|license|licens\w+|pay(s|ing)?|paid|charge[sd]?|margin|take rate|free to|premium|billing|bill for|revenue|arr|mrr|usage-based|per seat)\b",
 "team_not_stated": r"\b(founded by|co-?founders?|our founders?|our team|ex-[a-z]+|former(ly)?|previously|phd|worked at|alum(ni)?|we used to|my last role|we have been|years of experience|team of|2x founders|3x founders|spin-?out|spinoff|researchers? from|engineers from)\b",
}
from scipy.sparse import hstack, csr_matrix
def hand(ts):
    return np.array([[np.log1p(len(t.split())), float(bool(re.search(r"\d", t))), float(bool(re.search(r"[$€£]\s?\d", t)))] for t in ts])
ML = ["customer_unclear", "problem_unclear", "differentiation_missing"]
out = {"rules": {}, "ml": {}}
# rule evaluation: rule says "gap" when the pattern is NOT found
for g, pat in RULES.items():
    pred = np.array([0 if re.search(pat, t, re.I) else 1 for t in T]); y = df[g].values
    out["rules"][g] = {"prevalence": round(float(y.mean()), 3), "accuracy": round(accuracy_score(y, pred), 3),
                       "balanced_acc": round(balanced_accuracy_score(y, pred), 3),
                       "recall_of_gaps": round(recall_score(y, pred), 3), "precision_of_gaps": round(precision_score(y, pred), 3),
                       "majority_baseline": round(max(y.mean(), 1 - y.mean()), 3)}
# ML evaluation, TF-IDF only
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
mk = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
for g in ML:
    y = df[g].values; oof = np.zeros((3, len(y)))
    for i, (tr, te) in enumerate(rskf.split(T, y)):
        ttr, tte = [T[j] for j in tr], [T[j] for j in te]
        v = mk().fit(ttr); H = hand(ttr); mu, sd = H.mean(0), H.std(0) + 1e-6
        f = lambda ts: hstack([v.transform(ts), csr_matrix((hand(ts) - mu) / sd)]).tocsr()
        m = LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000).fit(f(ttr), y[tr])
        oof[i // 5, te] = m.predict_proba(f(tte))[:, 1]
    p = oof.mean(0); pred = (p >= .5).astype(int)
    out["ml"][g] = {"prevalence": round(float(y.mean()), 3), "AUC": round(float(np.mean([roc_auc_score(y, oof[r]) for r in range(3)])), 3),
                    "recall_of_gaps": round(recall_score(y, pred), 3), "precision_of_gaps": round(precision_score(y, pred), 3),
                    "balanced_acc": round(balanced_accuracy_score(y, pred), 3), "accuracy": round(accuracy_score(y, pred), 3),
                    "majority_baseline": round(max(y.mean(), 1 - y.mean()), 3)}
json.dump(out, open(f"{RES}/hybrid_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
# final fit on all 400, export for the browser
v = mk().fit(T); H = hand(T); MU, SD = H.mean(0), H.std(0) + 1e-6
X = hstack([v.transform(T), csr_matrix((H - MU) / SD)]).tocsr()
model = {"hand_mu": [float(x) for x in MU], "hand_sd": [float(x) for x in SD], "vocab": {t: [int(i), float(v.idf_[i])] for t, i in v.vocabulary_.items()}, "gaps": {}, "rules": RULES}
for g in ML:
    m = LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000).fit(X, df[g].values)
    model["gaps"][g] = {"coef": [round(float(c), 4) for c in m.coef_[0]], "intercept": round(float(m.intercept_[0]), 4)}
json.dump(model, open("app/readiness_model.json", "w"), separators=(",", ":"))
print("vocab", len(model["vocab"]), os.path.getsize("app/readiness_model.json") // 1024, "KB")
