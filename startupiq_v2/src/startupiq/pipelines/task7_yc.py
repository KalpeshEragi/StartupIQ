"""Task 7: text CNN and BiLSTM-attention vs the TF-IDF logistic regression, YC industry.
Run: PYTHONPATH=src python -m startupiq.pipelines.task7_yc
DECISION RULE (fixed before running): ship the simplest model (LR < CNN < BiLSTM) whose VALIDATION macro-F1
is at or above the lower 95% bootstrap bound of the best validation macro-F1. Test is scored once, afterwards."""
import json, os, pickle, time
import numpy as np, torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from startupiq.models.textnn import Vocab, TextCNN, BiLSTMAttn, predict_proba, train
from startupiq.pipelines.yc_split import load_split

SEED = 42
RES = "docs/results/task7"; INT = "data/interim/task7"
os.makedirs(RES, exist_ok=True); os.makedirs(INT, exist_ok=True)
torch.set_num_threads(2); T0 = time.time()
df, tr, va, te = load_split()
classes = sorted(df["label"].unique()); cid = {c: i for i, c in enumerate(classes)}
ytr, yva, yte = (np.array([cid[c] for c in d["label"]]) for d in (tr, va, te))
mf1 = lambda y, p: f1_score(y, p, average="macro")


def boot(y, p, fn, n=500):
    rng = np.random.default_rng(SEED); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); v.append(fn(y[i], p[i]))
    return [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]


voc = Vocab(tr["text"].tolist(), 2)
Xtr, Xva, Xte = (voc.encode(d["text"].tolist()) for d in (tr, va, te))
Ttr, Tva = torch.tensor(ytr), torch.tensor(yva)
cw = torch.tensor(len(ytr) / (len(classes) * np.bincount(ytr, minlength=len(classes))), dtype=torch.float)
V, C = len(voc.itos), len(classes)
out = {"vocab_size": V, "n_classes": C, "classes": classes, "sizes": {"train": len(tr), "val": len(va), "test": len(te)},
       "embeddings": "learned from scratch (pretrained vectors not available offline)",
       "decision_rule": "simplest model whose validation macro-F1 >= lower 95% bootstrap bound of the best validation macro-F1"}

# ---- reference: TF-IDF + logistic regression (Task 6 recipe) ----
t0 = time.time()
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english").fit(tr["text"])
lr = LogisticRegression(max_iter=2000, C=10, class_weight="balanced").fit(tf.transform(tr["text"]), ytr)
p_lr_va = lr.predict_proba(tf.transform(va["text"])); p_lr_te = lr.predict_proba(tf.transform(te["text"]))
res = {"LR": {"val_f1": mf1(yva, p_lr_va.argmax(1)), "fit_s": round(time.time() - t0, 1), "params": int(lr.coef_.size)}}
print("LR val", res["LR"], flush=True)

# ---- random search for CNN (seed 0, validation only) ----
rng = np.random.default_rng(SEED); trials = []
for t in range(4):
    cfg = dict(emb=int(rng.choice([50, 100, 200])), nf=int(rng.choice([32, 64, 128])), drop=float(rng.choice([0.3, 0.5])),
               lr=float(rng.choice([1e-3, 2e-3, 5e-3])), wd=float(rng.choice([1e-2, 1e-1])))
    cf = f"{INT}/trial{t}.json"
    if os.path.exists(cf): trials.append(json.load(open(cf))); continue
    if time.time() - T0 > 100: raise SystemExit("time budget: rerun to continue")
    torch.manual_seed(0); m = TextCNN(V, C, cfg["emb"], cfg["nf"], drop=cfg["drop"])
    f = train(m, Xtr, Ttr, Xva, yva, cw, 0, lr=cfg["lr"], wd=cfg["wd"])
    json.dump({**cfg, "val_f1": round(f, 4)}, open(cf, "w"))
    trials.append({**cfg, "val_f1": round(f, 4)}); print("trial", trials[-1], flush=True)
best_cfg = max(trials, key=lambda d: d["val_f1"]); out["cnn_random_search"] = trials; out["cnn_best_cfg"] = best_cfg


def run(kind, seed):
    torch.manual_seed(seed)
    if kind == "cnn":
        m = TextCNN(V, C, best_cfg["emb"], best_cfg["nf"], drop=best_cfg["drop"]); kw = dict(lr=best_cfg["lr"], wd=best_cfg["wd"])
    else:
        m = BiLSTMAttn(V, C); kw = dict(lr=2e-3, wd=1e-2)
    t = time.time(); f = train(m, Xtr, Ttr, Xva, yva, cw, seed, **kw)
    return m, f, time.time() - t


models = {}
for kind, seeds in (("cnn", range(3)), ("bilstm", range(2))):
    runs = []
    for s in seeds:
        ck = f"{INT}/{kind}_s{s}.pt"
        if os.path.exists(ck):
            m = TextCNN(V, C, best_cfg["emb"], best_cfg["nf"], drop=best_cfg["drop"]) if kind == "cnn" else BiLSTMAttn(V, C)
            m.load_state_dict(torch.load(ck)); f = mf1(yva, predict_proba(m, Xva).argmax(1)); secs = float(open(ck + ".s").read())
        else:
            if time.time() - T0 > 100: raise SystemExit("time budget: rerun to continue")
            m, f, secs = run(kind, s); torch.save(m.state_dict(), ck); open(ck + ".s", "w").write(str(secs))
        runs.append((f, s, m, secs)); print(kind, s, round(f, 4), flush=True)
    runs.sort(key=lambda r: -r[0]); f, s, m, secs = runs[0]
    models[kind] = m; torch.save(m.state_dict(), f"{INT}/{kind}.pt")
    res[kind.upper() if kind == "cnn" else "BiLSTM"] = {
        "val_f1": f, "val_f1_seeds": [round(r[0], 4) for r in runs], "val_f1_mean": float(np.mean([r[0] for r in runs])),
        "val_f1_sd": float(np.std([r[0] for r in runs])), "best_seed": s, "fit_s": round(secs, 1),
        "params": int(sum(p.numel() for p in m.parameters()))}
res["CNN"] = res.get("CNN"); 
pickle.dump({"vocab": voc, "classes": classes, "cfg": best_cfg}, open(f"{INT}/meta.pkl", "wb"))

# ---- decision on VALIDATION ----
pv = {"LR": p_lr_va.argmax(1), "CNN": predict_proba(models["cnn"], Xva).argmax(1), "BiLSTM": predict_proba(models["bilstm"], Xva).argmax(1)}
vf = {k: mf1(yva, v) for k, v in pv.items()}; best = max(vf, key=vf.get); lo = boot(yva, pv[best], mf1)[0]
order = ["LR", "CNN", "BiLSTM"]; chosen = next(k for k in order if vf[k] >= lo)
out["decision"] = {"val_macro_f1": {k: round(v, 4) for k, v in vf.items()}, "best_on_val": best, "lower_bound": lo, "shipped": chosen}
print("decision", out["decision"], flush=True)

# ---- test, scored once ----
pt = {"LR": p_lr_te, "CNN": predict_proba(models["cnn"], Xte), "BiLSTM": predict_proba(models["bilstm"], Xte)}
out["test"] = {}
for k in order:
    p = pt[k].argmax(1)
    out["test"][k] = {"accuracy": round(accuracy_score(yte, p), 3), "macro_f1": round(mf1(yte, p), 3),
                      "macro_f1_ci": boot(yte, p, mf1), "acc_ci": boot(yte, p, accuracy_score),
                      "val_f1": round(vf[k], 3), **{x: res[k][x] for x in ("fit_s", "params")}}
    print(k, out["test"][k], flush=True)
out["cnn_seed_spread"] = {k: res[k] for k in ("CNN", "BiLSTM")}
np.save(f"{INT}/test_proba_lr.npy", p_lr_te)
json.dump(out, open(f"{RES}/results_task7.json", "w"), indent=2, default=float)
