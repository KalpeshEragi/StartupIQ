"""Task 7 (competitor finder): a text CNN learns to predict the top-50 tags from a startup description (sigmoid, multi-label).
Its internal vector is then used to find similar startups (competitors). Compared with TF-IDF and LSA (SVD) similarity.
Run (repeat until it prints DONE; each call has a time limit): PYTHONPATH=src python -m startupiq.pipelines.task7_competitors

DECISION RULE (fixed before running): candidates ordered simplest to most complex: TFIDF, LSA, CNN, HYBRID.
On VALIDATION queries (corpus = train), ship the simplest whose precision@10 is at or above the lower 95% bootstrap bound of the best.
Precision@10 = share of the 10 retrieved startups that share at least one (non-industry) tag with the query.
Test queries (corpus = train + val) are scored once, after the decision."""
import json, os, time
import numpy as np, torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import normalize
from startupiq.models.textnn import Vocab, TextCNN, predict_sigmoid, train_multilabel
from startupiq.pipelines.comp_common import prep, relevance, cnn_vectors, evaluate, boot_ci, paired_ci, masks

T0 = time.time(); INT = "data/interim/task7c"; RES = "docs/results/task7_competitors"
os.makedirs(INT, exist_ok=True); os.makedirs(RES, exist_ok=True); torch.set_num_threads(2)
D = prep(); tr, va, te = D["tr"], D["va"], D["te"]
voc = Vocab(tr["text"].tolist(), 2); V = len(voc.itos); C = len(D["top_tags"])
Xtr, Xva, Xte = (voc.encode(d["text"].tolist()) for d in (tr, va, te))
keep_tr, keep_va = D["Y_tr"].sum(1) > 0, D["Y_va"].sum(1) > 0
CFG = dict(emb=100, nf=128, drop=0.5, lr=0.005, wd=0.1)       # best setting from the Task 7 industry search
pos = D["Y_tr"][keep_tr].sum(0).clip(1); pw = torch.tensor(np.clip((keep_tr.sum() - pos) / pos, 1, 20), dtype=torch.float)


def budget():
    if time.time() - T0 > 90: raise SystemExit("time budget reached: rerun")


# ---- 1. train CNN on tags (2 seeds, cached) ----
cnns = {}
for seed in (0, 1):
    ck = f"{INT}/cnn_s{seed}.pt"
    m = TextCNN(V, C, CFG["emb"], CFG["nf"], drop=CFG["drop"])
    if os.path.exists(ck):
        m.load_state_dict(torch.load(ck))
    else:
        budget(); torch.manual_seed(seed); t = time.time()
        f = train_multilabel(m, Xtr[keep_tr], torch.tensor(D["Y_tr"][keep_tr]), Xva[keep_va], torch.tensor(D["Y_va"][keep_va]), pw, seed, lr=CFG["lr"], wd=CFG["wd"])
        torch.save(m.state_dict(), ck); json.dump({"val_ap": f, "secs": time.time() - t}, open(ck + ".json", "w")); print("trained seed", seed, f, flush=True)
    m.eval(); cnns[seed] = m
info = {s: json.load(open(f"{INT}/cnn_s{s}.pt.json")) for s in cnns}
best_seed = max(info, key=lambda s: info[s]["val_ap"]); cnn = cnns[best_seed]

# ---- 2. tag prediction: CNN vs TF-IDF one-vs-rest logistic regression (test) ----
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english").fit(tr["text"])
Wtr, Wva, Wte = (tf.transform(d["text"]) for d in (tr, va, te))
pf = f"{INT}/lr_tags_te.npy"
if not os.path.exists(pf):
    budget(); P = np.zeros((len(te), C))
    for j in range(C):
        y = D["Y_tr"][:, j]
        P[:, j] = LogisticRegression(max_iter=300, C=5, class_weight="balanced").fit(Wtr, y).predict_proba(Wte)[:, 1]
    np.save(pf, P)
P_lr = np.load(pf); P_cnn = predict_sigmoid(cnn, Xte); Yte = D["Y_te"]
ok = [j for j in range(C) if 0 < Yte[:, j].sum() < len(Yte)]
tagres = {n: {"micro_AP": round(float(average_precision_score(Yte, P, average="micro")), 3),
              "macro_AUC": round(float(np.mean([roc_auc_score(Yte[:, j], P[:, j]) for j in ok])), 3)} for n, P in (("TFIDF_LR", P_lr), ("CNN", P_cnn))}
tagres["random_micro_AP"] = round(float(Yte.mean()), 3); print("tags", tagres, flush=True)

# ---- 3. retrieval: build vector spaces (fit on train only) ----
svd = TruncatedSVD(200, random_state=42).fit(Wtr)
def spaces(Wm, X):
    cv, mu = None, None
    return {"TFIDF": Wm, "LSA": normalize(svd.transform(Wm))}
def cnn_space(X, mu=None): return cnn_vectors(cnn, X, mu)
Ctr, mu = cnn_space(Xtr); Cva, _ = cnn_space(Xva, mu); Cte, _ = cnn_space(Xte, mu)
SP = {"tr": {"TFIDF": Wtr, "LSA": normalize(svd.transform(Wtr)), "CNN": Ctr}, "va": {"TFIDF": Wva, "LSA": normalize(svd.transform(Wva)), "CNN": Cva},
      "te": {"TFIDF": Wte, "LSA": normalize(svd.transform(Wte)), "CNN": Cte}}
def sim(q, c, name):
    if name == "HYBRID":
        z = lambda S: (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
        return 0.5 * z(sim(q, c, "TFIDF")) + 0.5 * z(sim(q, c, "CNN"))
    A, B = SP[q][name], SP[c][name]
    S = A @ B.T
    return S.toarray() if hasattr(S, "toarray") else np.asarray(S)

names = ["TFIDF", "LSA", "CNN", "HYBRID"]; out = {"tags_test": tagres, "cnn_val_ap": {s: info[s]["val_ap"] for s in info}, "best_seed": best_seed, "cfg": CFG, "n_tags": C}
# validation (decision): queries = val, corpus = train
rel_v = relevance(D["T_va"], D["T_tr"]); ind_v, dup_v = masks(va, tr); valid_v = D["T_va"].sum(1) > 0
val = {n: evaluate(sim("va", "tr", n), rel_v, ind_v, valid_v, dup_v) for n in names}
best = max(names, key=lambda n: val[n]["p10"].mean()); lo = boot_ci(val[best]["p10"])[0]
chosen = next(n for n in names if val[n]["p10"].mean() >= lo)
out["decision"] = {"val_p10": {n: round(float(val[n]["p10"].mean()), 3) for n in names}, "best_on_val": best, "lower_bound": lo, "shipped": chosen,
                   "random_p10": round(float(rel_v[valid_v].mean()), 3)}
print("decision", out["decision"], flush=True)
# test (once): corpus = train + val
import pandas as pd
from scipy.sparse import vstack as svstack
cor = pd.concat([tr, va])
SPc = {"TFIDF": svstack([SP["tr"]["TFIDF"], SP["va"]["TFIDF"]]).tocsr(), "LSA": np.vstack([SP["tr"]["LSA"], SP["va"]["LSA"]]),
       "CNN": np.vstack([SP["tr"]["CNN"], SP["va"]["CNN"]])}
SP["co"] = SPc; Tco = np.vstack([D["T_tr"], D["T_va"]])
rel_t = relevance(D["T_te"], Tco); ind_t, dup_t = masks(te, cor); valid_t = D["T_te"].sum(1) > 0
test = {n: evaluate(sim("te", "co", n), rel_t, ind_t, valid_t, dup_t) for n in names}
out["test"] = {n: {"p10": round(float(test[n]["p10"].mean()), 3), "p10_ci": boot_ci(test[n]["p10"]), "p1": round(float(test[n]["p1"].mean()), 3),
                   "same_industry_at_10": round(float(test[n]["ind10"].mean()), 3),
                   "p10_minus_TFIDF_ci": paired_ci(test[n]["p10"], test["TFIDF"]["p10"]) if n != "TFIDF" else None} for n in names}
out["test"]["random_p10"] = round(float(rel_t[valid_t].mean()), 3); out["n_test_queries"] = int(valid_t.sum())
out["cnn_params"] = int(sum(p.numel() for p in cnn.parameters())); out["embedding_dim"] = int(Ctr.shape[1])
json.dump(out, open(f"{RES}/results_task7_competitors.json", "w"), indent=2, default=float)
print(json.dumps(out["test"], indent=1)); print("DONE")
