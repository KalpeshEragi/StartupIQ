"""Task 8: interpretability and reliability for the YC industry models (LR, text CNN, BiLSTM-attention).
Run after task7: PYTHONPATH=src python -m startupiq.pipelines.task8_yc
Everything is computed on the test split using models trained/selected in Task 7 (no refitting on test)."""
import json, os, pickle, re
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from startupiq.models.textnn import TextCNN, BiLSTMAttn, predict_proba, tokenize
from startupiq.pipelines.yc_split import load_split

RES = "docs/results/task8"; FIG = RES + "/figures"; INT = "data/interim/task7"
os.makedirs(FIG, exist_ok=True); torch.set_num_threads(2)
rng = np.random.default_rng(42)
meta = pickle.load(open(f"{INT}/meta.pkl", "rb")); voc, classes, cfg = meta["vocab"], meta["classes"], meta["cfg"]
df, tr, va, te = load_split(); cid = {c: i for i, c in enumerate(classes)}
yte = np.array([cid[c] for c in te["label"]]); ytr = np.array([cid[c] for c in tr["label"]]); C = len(classes); V = len(voc.itos)
cnn = TextCNN(V, C, cfg["emb"], cfg["nf"], drop=cfg["drop"]); cnn.load_state_dict(torch.load(f"{INT}/cnn.pt")); cnn.eval()
lstm = BiLSTMAttn(V, C); lstm.load_state_dict(torch.load(f"{INT}/bilstm.pt")); lstm.eval()
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english").fit(tr["text"])
lr = LogisticRegression(max_iter=2000, C=10, class_weight="balanced").fit(tf.transform(tr["text"]), ytr)
texts = te["text"].tolist()
P = {"LR": lambda T: lr.predict_proba(tf.transform(T)),
     "CNN": lambda T: predict_proba(cnn, voc.encode(T)),
     "BiLSTM": lambda T: predict_proba(lstm, voc.encode(T))}
mf1 = lambda y, p: f1_score(y, p, average="macro")
out = {}

# ---- 1. Linear model: top terms per class ----
fn = np.array(tf.get_feature_names_out())
out["lr_top_terms"] = {c: fn[np.argsort(-lr.coef_[i])[:8]].tolist() for i, c in enumerate(classes)}

# ---- 2. CNN: gradient x input saliency, and filter n-gram contributions ----
def cnn_saliency(text):
    x = voc.encode([text]); e = cnn.emb(x).detach().requires_grad_(True)
    lg = cnn(x, e=e); k = int(lg.argmax()); lg[0, k].backward()
    s = (e.grad * e).sum(-1).abs()[0].detach().numpy()
    n = min(len(tokenize(text)), x.shape[1]); return s[:n], k

def cnn_ngrams(text, top=3):
    """Per-prediction n-gram contributions: for each filter, value at its max position x output weight for the predicted class."""
    x = voc.encode([text]); n = min(len(tokenize(text)), x.shape[1]); toks = tokenize(text)[:n]
    with torch.no_grad():
        e = cnn.emb(x); h = e.transpose(1, 2); k = int(cnn(x).argmax()); contribs = []
        for ci, conv in enumerate(cnn.convs):
            a = torch.relu(conv(h))[0]; m, pos = a.max(dim=1); w = cnn.fc.weight[k, ci * cnn.nf:(ci + 1) * cnn.nf]
            for f in range(cnn.nf):
                contribs.append(((m[f] * w[f]).item(), " ".join(toks[pos[f]:pos[f] + cnn.ks[ci]])))
    agg = {}
    for v, g in contribs: agg[g] = agg.get(g, 0) + v
    return sorted(agg.items(), key=lambda t: -t[1])[:top]

# ---- 3. Token-erasure importance (model-agnostic) vs attention / saliency ----
def erasure(model_key, text):
    toks = tokenize(text)[:100]; base = P[model_key]([text])[0]; k = int(base.argmax())
    alts = [" ".join(toks[:i] + toks[i + 1:]) for i in range(len(toks))]
    return base[k] - P[model_key](alts)[:, k] if alts else np.zeros(0)

idx = rng.choice(len(texts), 150, replace=False)
rho_att, rho_sal, ex = [], [], []
for i in idx:
    t = texts[i]; er = erasure("BiLSTM", t); n = len(er)
    if n < 5: continue
    with torch.no_grad(): _, a = lstm(voc.encode([t]), return_attn=True)
    r = spearmanr(a[0, :n].numpy(), er).statistic
    if not np.isnan(r): rho_att.append(r)
    sal, _ = cnn_saliency(t); er2 = erasure("CNN", t)
    r2 = spearmanr(sal[:len(er2)], er2).statistic
    if not np.isnan(r2): rho_sal.append(r2)
out["attention_vs_erasure_spearman"] = {"mean": round(float(np.mean(rho_att)), 3), "median": round(float(np.median(rho_att)), 3), "n": len(rho_att)}
out["cnn_saliency_vs_erasure_spearman"] = {"mean": round(float(np.mean(rho_sal)), 3), "median": round(float(np.median(rho_sal)), 3), "n": len(rho_sal)}

# ---- 4. Worked examples (for the 30-explanation hand-rating sheet) ----
pc = P["CNN"](texts).argmax(1); pl = P["LR"](texts).argmax(1)
rows = []
for i in rng.choice(len(texts), 30, replace=False):
    t = texts[i]; sal, k = cnn_saliency(t); tk = tokenize(t)[:len(sal)]
    top_tok = [tk[j] for j in np.argsort(-sal)[:5]]
    lrk = pl[i]; contrib = tf.transform([t]).toarray()[0] * lr.coef_[lrk]
    rows.append({"text": t[:220], "true": classes[yte[i]], "cnn_pred": classes[pc[i]], "lr_pred": classes[lrk],
                 "cnn_saliency_tokens": top_tok, "cnn_ngrams": [g for g, _ in cnn_ngrams(t)],
                 "lr_terms": fn[np.argsort(-contrib)[:5]].tolist(), "rating_1to5": None})
pd.DataFrame(rows).to_csv(f"{RES}/explanations_to_rate.csv", index=False)
out["explanations_file"] = "explanations_to_rate.csv (30 rows, rating column left empty for a human)"

# ---- 5. Perturbation tests ----
sector = set(w for c in classes for w in fn[np.argsort(-lr.coef_[cid[c]])[:25]] for w in w.split())
def typo(t):
    ws = t.split()
    for j in rng.choice(len(ws), max(1, len(ws) // 10), replace=False):
        w = ws[j]
        if len(w) > 3: k = rng.integers(1, len(w) - 2); ws[j] = w[:k] + w[k + 1] + w[k] + w[k + 2:]
    return " ".join(ws)
def mask_top(model_key):
    def f(t):
        er = erasure(model_key, t); tk = t.split()
        if len(er) == 0: return t
        top = {tokenize(t)[j] for j in np.argsort(-er)[:3] if j < len(tokenize(t))}
        return " ".join(w for w in tk if re.sub(r"[^a-z0-9']", "", w.lower()) not in top)
    return f
def first_sentence(t): return re.split(r"(?<=[.!?])\s", t)[0]
def drop_sector(t): return " ".join(w for w in t.split() if re.sub(r"[^a-z0-9']", "", w.lower()) not in sector)
sub = rng.choice(len(texts), 300, replace=False); st = [texts[i] for i in sub]; sy = yte[sub]
pert = {"typos (10% of words)": lambda m: [typo(t) for t in st], "first sentence only": lambda m: [first_sentence(t) for t in st],
        "remove sector words": lambda m: [drop_sector(t) for t in st], "mask top-3 words": lambda m: [mask_top(m)(t) for t in st]}
out["perturbation_macro_f1"] = {}
for m in P:
    base = mf1(sy, P[m](st).argmax(1)); row = {"unperturbed": round(base, 3)}
    for name, mk in pert.items(): row[name] = round(mf1(sy, P[m](mk(m)).argmax(1)), 3)
    out["perturbation_macro_f1"][m] = row; print(m, row, flush=True)

# ---- 6. Reliability: ECE, accuracy by length and batch year, out-of-domain text ----
def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1); e = 0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)
probs = {m: P[m](texts) for m in P}
out["ece"] = {m: round(ece(probs[m], yte), 3) for m in P}
wl = te["text_words"].values; q = np.quantile(wl, [1 / 3, 2 / 3]); grp = np.digitize(wl, q)
out["accuracy_by_length"] = {m: {n: round(float((probs[m].argmax(1)[grp == g] == yte[grp == g]).mean()), 3) for g, n in enumerate(["short", "medium", "long"])} for m in P}
yr = te["batch_year"].values; old = yr < 2025
out["accuracy_by_batch_era"] = {m: {"before_2025": round(float((probs[m].argmax(1)[old] == yte[old]).mean()), 3),
                                    "2025_on": round(float((probs[m].argmax(1)[~old] == yte[~old]).mean()), 3)} for m in P}
ood = ["We sell second hand cars and run a local garage with six mechanics.", "Our bakery makes sourdough and pastries for cafes in the city centre.",
       "The poem explores memory and loss through the voice of a river.", "Weekly yoga classes and meditation retreats for busy parents."]
out["out_of_domain_max_confidence"] = {m: [round(float(x), 2) for x in P[m](ood).max(1)] for m in P}
out["in_domain_mean_max_confidence"] = {m: round(float(probs[m].max(1).mean()), 3) for m in P}
thr = 0.6; out["abstain_at_0.6"] = {}
for m in P:
    keep = probs[m].max(1) >= thr
    out["abstain_at_0.6"][m] = {"coverage": round(float(keep.mean()), 3), "accuracy_on_kept": round(float((probs[m].argmax(1)[keep] == yte[keep]).mean()), 3)}

# ---- 7. Comparison table + plain figures ----
t7 = json.load(open("docs/results/task7/results_task7.json"))["test"]
out["comparison"] = {m: {"test_macro_f1": t7[m]["macro_f1"], "test_acc": t7[m]["accuracy"], "params": t7[m]["params"], "fit_s": t7[m]["fit_s"],
                         "ece": out["ece"][m], "typo_macro_f1": out["perturbation_macro_f1"][m]["typos (10% of words)"]} for m in P}
json.dump(out, open(f"{RES}/results_task8.json", "w"), indent=2, default=float)

fig, ax = plt.subplots(figsize=(6, 3.5)); ms = list(P); ks = list(pert)
w = 0.15; ax.bar(np.arange(len(ms)) - 2 * w, [out["perturbation_macro_f1"][m]["unperturbed"] for m in ms], w, label="unperturbed", color="#444")
for j, k in enumerate(ks): ax.bar(np.arange(len(ms)) + (j - 1) * w, [out["perturbation_macro_f1"][m][k] for m in ms], w, label=k)
ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms); ax.set_ylabel("macro-F1"); ax.set_title("Perturbation tests (300 test texts)"); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(f"{FIG}/t8_perturbation.png", dpi=120); plt.close(fig)
fig, ax = plt.subplots(figsize=(4.5, 4))
for m in P:
    conf = probs[m].max(1); ok = probs[m].argmax(1) == yte; b = np.linspace(0, 1, 11); xs, ys = [], []
    for lo in b[:-1]:
        s = (conf > lo) & (conf <= lo + 0.1)
        if s.sum() >= 10: xs.append(conf[s].mean()); ys.append(ok[s].mean())
    ax.plot(xs, ys, marker="o", label=m)
ax.plot([0, 1], [0, 1], "--", color="gray"); ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_title("Calibration (test)"); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/t8_calibration.png", dpi=120); plt.close(fig)
print(json.dumps(out["comparison"], indent=1))
