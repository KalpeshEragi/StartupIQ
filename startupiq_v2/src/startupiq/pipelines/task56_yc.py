"""Tasks 5 and 6 on Stream A (YC text): features, selection, splits, classical models, clustering.
Run: PYTHONPATH=src python -m startupiq.pipelines.task56_yc
Rules: vectorisers fitted on train only; model chosen on validation; test scored once."""
import json, os, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, confusion_matrix, f1_score,
                             normalized_mutual_info_score, silhouette_score, top_k_accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

SEED = 42
RES = "docs/results/task5_6"; FIG = os.path.join(RES, "figures")
BLUE, GRAY, INK = "#2a78d6", "#b8bcc2", "#222222"
os.makedirs(FIG, exist_ok=True)
out = {}


def ci(y, p, fn, n=500):
    rng = np.random.default_rng(SEED); y = np.asarray(y); p = np.asarray(p); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); v.append(fn(y[i], p[i]))
    return [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]


mf1 = lambda y, p: f1_score(y, p, average="macro")

# ---------------- Task 5: data, leakage exclusions, split ----------------
df = pd.read_csv("data/processed/yc_clean.csv")
df = df[df["usable_text"] & df["industry"].notna()].reset_index(drop=True)
classes = sorted(df["industry"].unique())
# classes with < 30 companies cannot be learned or evaluated reliably: merge into "Other"
small = df["industry"].value_counts()[lambda s: s < 50].index.tolist()
df["label"] = df["industry"].where(~df["industry"].isin(small), "Other (merged)")
out["merged_small_classes"] = small
out["n_rows"] = len(df); out["classes"] = df["label"].value_counts().to_dict()
EXCLUDED = {"tags": "five tags are literally industry names", "subindustry/industries": "restate the label",
            "status, isHiring": "crawl-time snapshots", "founder flags": "audit only", "name": "identifier",
            "website/domain": "identifier"}
out["excluded_inputs"] = EXCLUDED

tr_va, te = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=SEED)
tr, va = train_test_split(tr_va, test_size=0.1765, stratify=tr_va["label"], random_state=SEED)
out["split_sizes"] = {"train": len(tr), "val": len(va), "test": len(te)}
# temporal holdout: older batches train, newest test
tmask = df["batch_year"] >= 2025
out["temporal_split"] = {"train": int((~tmask).sum()), "test": int(tmask.sum())}

wordv = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
charv = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=60000)


def struct(d):
    return csr_matrix(np.c_[d["team_size_log"].fillna(d["team_size_log"].median() if False else 0).values * 0,
                            d["team_size_missing"].astype(float).values, (d["stage"] == "Growth").astype(float).values])


class Feat:
    """All fitted state lives here and is fitted on train only."""
    def __init__(self, kind): self.kind = kind
    def fit(self, d):
        self.w = wordv().fit(d["text"]); self.c = charv().fit(d["text"]) if "char" in self.kind else None
        self.med = d["team_size_log"].median()
        self.mu, self.sd = d["team_size_log"].fillna(self.med).mean(), d["team_size_log"].fillna(self.med).std()
        return self
    def transform(self, d):
        parts = [self.w.transform(d["text"])]
        if self.c is not None: parts.append(self.c.transform(d["text"]))
        if "struct" in self.kind:
            z = (d["team_size_log"].fillna(self.med) - self.mu) / self.sd
            parts.append(csr_matrix(np.c_[z, d["team_size_missing"].astype(float), (d["stage"] == "Growth").astype(float)]))
        return hstack(parts).tocsr()


def lr(): return LogisticRegression(max_iter=2000, C=10, class_weight="balanced")


# feature-set ablation (validation macro-F1)
abl = {}
for kind in ["word", "word+char", "word+struct", "word+char+struct"]:
    f = Feat(kind).fit(tr); m = lr().fit(f.transform(tr), tr["label"]); p = m.predict(f.transform(va))
    abl[kind] = {"val_macro_f1": round(mf1(va["label"], p), 3), "val_acc": round(accuracy_score(va["label"], p), 3),
                 "n_features": f.transform(tr).shape[1]}
out["feature_ablation"] = abl
best_kind = max(abl, key=lambda k: abl[k]["val_macro_f1"]); out["best_feature_set"] = best_kind
F = Feat(best_kind).fit(tr); Xtr, Xva, Xte = F.transform(tr), F.transform(va), F.transform(te)

# feature selection: chi2 top-k vs none
sel = {}
for k in [500, 1000, 2000, 5000, 10000, Xtr.shape[1]]:
    s = SelectKBest(chi2, k=min(k, Xtr.shape[1])).fit(Xtr, tr["label"])
    p = lr().fit(s.transform(Xtr), tr["label"]).predict(s.transform(Xva))
    sel[int(min(k, Xtr.shape[1]))] = round(mf1(va["label"], p), 3)
out["chi2_selection_val_macro_f1"] = sel
# L1 as a second selector
l1 = LogisticRegression(penalty="l1", solver="saga", C=2, max_iter=300, class_weight="balanced").fit(Xtr, tr["label"])
nz = int((np.abs(l1.coef_).sum(0) > 0).sum())
p = l1.predict(Xva); out["l1_selection"] = {"nonzero_features": nz, "val_macro_f1": round(mf1(va["label"], p), 3)}
bestk = max(sel, key=sel.get)
SEL = SelectKBest(chi2, k=bestk).fit(Xtr, tr["label"]) if sel[bestk] > sel[Xtr.shape[1]] + 0.005 else None
out["selection_used"] = int(bestk) if SEL is not None else "none (did not help)"
T = (lambda X: SEL.transform(X)) if SEL is not None else (lambda X: X)
Xtr2, Xva2, Xte2 = T(Xtr), T(Xva), T(Xte)

# ---------------- Task 6: models ----------------
svd = TruncatedSVD(200, random_state=SEED).fit(Xtr2)
Str, Sva, Ste = svd.transform(Xtr2), svd.transform(Xva2), svd.transform(Xte2)
models = {
    "majority baseline": (DummyClassifier(strategy="most_frequent"), False),
    "logistic regression (TF-IDF)": (lr(), False),
    "linear SVM (TF-IDF)": (LinearSVC(C=0.5, class_weight="balanced"), False),
    "complement naive Bayes": (ComplementNB(alpha=0.3), False),
    "random forest (SVD-200)": (RandomForestClassifier(400, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED), True),
}
res = {}; fitted = {}
for name, (m, dense) in models.items():
    t = time.time(); m.fit(Str if dense else Xtr2, tr["label"])
    p = m.predict(Sva if dense else Xva2); fitted[name] = m
    res[name] = {"val_acc": round(accuracy_score(va["label"], p), 3), "val_macro_f1": round(mf1(va["label"], p), 3),
                 "fit_sec": round(time.time() - t, 1)}
vote = VotingClassifier([("lr", lr()), ("nb", ComplementNB(alpha=0.3))], voting="soft").fit(Xtr2, tr["label"])
p = vote.predict(Xva2); fitted["soft-voting (LR + NB)"] = vote
res["soft-voting (LR + NB)"] = {"val_acc": round(accuracy_score(va["label"], p), 3), "val_macro_f1": round(mf1(va["label"], p), 3), "fit_sec": 0}
out["model_validation"] = res
cand = {k: v for k, v in res.items() if k != "majority baseline"}
best = max(cand, key=lambda k: cand[k]["val_macro_f1"]); out["chosen_model"] = best

# test scored once for the chosen model and the baseline
def test_eval(name):
    m = fitted[name]; dense = name.startswith("random forest")
    X = Ste if dense else Xte2; p = m.predict(X)
    r = {"acc": round(accuracy_score(te["label"], p), 3), "acc_ci": ci(te["label"], p, accuracy_score),
         "macro_f1": round(mf1(te["label"], p), 3), "macro_f1_ci": ci(te["label"], p, mf1)}
    if hasattr(m, "predict_proba"):
        pr = m.predict_proba(X); r["top3_acc"] = round(top_k_accuracy_score(te["label"], pr, k=3, labels=m.classes_), 3)
    return r, p
out["test_chosen"], pbest = test_eval(best)
out["test_baseline"], _ = test_eval("majority baseline")

# top-3 for the product: calibrated linear SVM on the same features
cal = CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=5).fit(Xtr2, tr["label"])
pr = cal.predict_proba(Xva2); out["val_top3_calibrated_svm"] = round(top_k_accuracy_score(va["label"], pr, k=3, labels=cal.classes_), 3)
pr_te = cal.predict_proba(Xte2)
out["test_calibrated_svm"] = {"acc": round(accuracy_score(te["label"], cal.classes_[pr_te.argmax(1)]), 3),
    "top3_acc": round(top_k_accuracy_score(te["label"], pr_te, k=3, labels=cal.classes_), 3)}
conf = pr_te.max(1); pred = cal.classes_[pr_te.argmax(1)]
for th in (0.5, 0.6, 0.7):
    keep = conf >= th
    out[f"abstain_at_{th}"] = {"coverage": round(float(keep.mean()), 3), "acc_on_answered": round(float(accuracy_score(te["label"][keep], pred[keep])), 3)}

# temporal check with the chosen feature recipe
trt, tet = df[~tmask], df[tmask]
Ft = Feat(best_kind).fit(trt); mt = lr().fit(Ft.transform(trt), trt["label"]); pt = mt.predict(Ft.transform(tet))
out["temporal_holdout_lr"] = {"acc": round(accuracy_score(tet["label"], pt), 3), "macro_f1": round(mf1(tet["label"], pt), 3), "n_test": len(tet)}

# ---------------- clustering on SVD embeddings (train+val+test text, no labels used) ----------------
Fall = Feat("word").fit(df); Sall = TruncatedSVD(100, random_state=SEED).fit_transform(Fall.transform(df))
Sall = Sall / np.linalg.norm(Sall, axis=1, keepdims=True).clip(1e-9)
clu = {}
for k in (6, 9, 12, 20):
    km = KMeans(k, n_init=5, random_state=SEED).fit(Sall); ag = AgglomerativeClustering(k, linkage="ward").fit(Sall)
    for nm, lab in (("kmeans", km.labels_), ("ward", ag.labels_)):
        clu[f"{nm}_k{k}"] = {"silhouette": round(float(silhouette_score(Sall, lab, sample_size=3000, random_state=SEED)), 3),
                             "NMI_vs_industry": round(normalized_mutual_info_score(df["label"], lab), 3),
                             "ARI_vs_industry": round(adjusted_rand_score(df["label"], lab), 3)}
out["clustering"] = clu

# ---------------- figures ----------------
def sty(ax, t):
    ax.set_title(t, loc="left", fontsize=12, color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

# T5-1 feature-set ablation + selection curve
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
ks = list(abl); axs[0].barh(ks[::-1], [abl[k]["val_macro_f1"] for k in ks][::-1], color=BLUE)
for i, k in enumerate(ks[::-1]): axs[0].text(abl[k]["val_macro_f1"], i, f" {abl[k]['val_macro_f1']:.3f}", va="center")
sty(axs[0], "Feature sets (validation macro-F1)")
axs[1].plot(list(sel), list(sel.values()), "o-", color=BLUE); axs[1].set_xscale("log")
axs[1].set_xlabel("features kept (chi-squared)"); axs[1].set_ylabel("validation macro-F1"); sty(axs[1], "Does feature selection help?")
fig.tight_layout(); fig.savefig(f"{FIG}/t5_features_selection.png", dpi=150); plt.close(fig)

# T5-2 exclusions + split table
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
axs[0].axis("off")
t = axs[0].table(cellText=[[k, v] for k, v in EXCLUDED.items()], colLabels=["input excluded from model", "why (leakage control)"], loc="center", cellLoc="left", colWidths=[0.35, 0.65])
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.6); axs[0].set_title("Leakage exclusions", loc="left")
sz = out["split_sizes"]; axs[1].barh(["test (scored once)", "validation", "train"], [sz["test"], sz["val"], sz["train"]], color=[GRAY, GRAY, BLUE])
for i, v in enumerate([sz["test"], sz["val"], sz["train"]]): axs[1].text(v, i, f" {v}", va="center")
sty(axs[1], "Stratified split (70 / 15 / 15)")
fig.tight_layout(); fig.savefig(f"{FIG}/t5_leakage_split.png", dpi=150); plt.close(fig)

# T6-1 model comparison with test CI for chosen
names = list(res); fig, ax = plt.subplots(figsize=(10, 4.8))
ax.barh(names[::-1], [res[n]["val_macro_f1"] for n in names][::-1], color=[BLUE if n == best else GRAY for n in names][::-1])
for i, n in enumerate(names[::-1]): ax.text(res[n]["val_macro_f1"], i, f" F1 {res[n]['val_macro_f1']:.3f} | acc {res[n]['val_acc']:.3f}", va="center", fontsize=9)
ax.set_xlim(0, 1); sty(ax, f"Model comparison on validation (chosen: {best})"); fig.tight_layout(); fig.savefig(f"{FIG}/t6_model_comparison.png", dpi=150); plt.close(fig)

# T6-2 confusion matrix
labs = sorted(df["label"].unique()); cm = confusion_matrix(te["label"], pbest, labels=labs); cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(8.5, 7)); im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=8); ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8)
for i in range(len(labs)):
    for j in range(len(labs)): ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7, color="white" if cmn[i, j] > .5 else INK)
ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(f"Test confusion matrix, {best} (row-normalised)", loc="left"); fig.colorbar(im, ax=ax)
fig.tight_layout(); fig.savefig(f"{FIG}/t6_confusion_matrix.png", dpi=150); plt.close(fig)

# T6-3 market map: SVD -> 2D (PCA of embedding), 3 colours + gray
from sklearn.decomposition import PCA
xy = PCA(2, random_state=SEED).fit_transform(Sall); top3 = df["label"].value_counts().index[:3].tolist()
fig, ax = plt.subplots(figsize=(8.5, 7)); g = ~df["label"].isin(top3); ax.scatter(xy[g, 0], xy[g, 1], s=5, c=GRAY, label="other sectors")
for c, col in zip(top3, ["#2a78d6", "#e8743b", "#2fa36b"]):
    m = (df["label"] == c).values; ax.scatter(xy[m, 0], xy[m, 1], s=6, c=col, label=c, alpha=.7)
ax.legend(markerscale=3); sty(ax, "Text-embedding map of YC companies (2D projection)"); fig.tight_layout(); fig.savefig(f"{FIG}/t6_market_map.png", dpi=150); plt.close(fig)

json.dump(out, open(os.path.join(RES, "results_yc.json"), "w"), indent=1, default=str)
print(json.dumps({k: out[k] for k in ["best_feature_set", "selection_used", "chosen_model", "test_chosen", "test_baseline", "test_calibrated_svm", "temporal_holdout_lr", "abstain_at_0.6"]}, indent=1))
print(abl); print(sel); print(res)
