"""Task 8 (competitor finder): why was each competitor matched, is the explanation faithful, and when is the finder unreliable?
Run after task7_competitors: PYTHONPATH=src python -m startupiq.pipelines.task8_competitors
Shipped retriever from Task 7's decision rule is LSA (TF-IDF -> SVD 200); the CNN embedding is explained too, for comparison."""
import json, os, re, time
import numpy as np, pandas as pd, torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from startupiq.models.textnn import Vocab, TextCNN
from startupiq.pipelines.comp_common import prep, relevance, cnn_vectors, evaluate, masks, boot_ci

RES = "docs/results/task8_competitors"; INT = "data/interim/task7c"; os.makedirs(RES, exist_ok=True); torch.set_num_threads(2)
rng = np.random.default_rng(42)
D = prep(); tr, va, te = D["tr"], D["va"], D["te"]; cor = pd.concat([tr, va]).reset_index(drop=True); Tco = np.vstack([D["T_tr"], D["T_va"]])
t7 = json.load(open("docs/results/task7_competitors/results_task7_competitors.json"))
voc = Vocab(tr["text"].tolist(), 2); V = len(voc.itos); C = t7["n_tags"]; cfg = t7["cfg"]
cnn = TextCNN(V, C, cfg["emb"], cfg["nf"], drop=cfg["drop"]); cnn.load_state_dict(torch.load(f"{INT}/cnn_s{t7['best_seed']}.pt")); cnn.eval()
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english").fit(tr["text"])
svd = TruncatedSVD(200, random_state=42).fit(tf.transform(tr["text"]))
_, mu = cnn_vectors(cnn, voc.encode(tr["text"].tolist()))
ENC = {"LSA": lambda T: normalize(svd.transform(tf.transform(list(T)))), "CNN": lambda T: cnn_vectors(cnn, voc.encode(list(T)), mu)[0]}
CV = {k: f(cor["text"].tolist()) for k, f in ENC.items()}
rel_t = relevance(D["T_te"], Tco); ind_t, dup_t = masks(te, cor); valid_t = D["T_te"].sum(1) > 0
texts = te["text"].tolist(); out = {}


def search(enc, T, k=10, dup=None):
    S = enc @ CV[enc_name].T
    if dup is not None: S[dup] = -1e9
    return np.argsort(-S, axis=1)[:, :k], S


def occlusion(enc_name, text, nb_vec):
    """Drop in similarity to one neighbour when each word is deleted. Big drop = that word is why they matched."""
    w = text.split()[:100]; base = float(ENC[enc_name]([" ".join(w)])[0] @ nb_vec)
    alts = [" ".join(w[:i] + w[i + 1:]) for i in range(len(w))]
    return w, base, base - (ENC[enc_name](alts) @ nb_vec)


# ---- 1. worked examples + hand-rating sheet (LSA, the shipped finder) ----
enc_name = "LSA"; Q = ENC["LSA"](texts); S = Q @ CV["LSA"].T; S[dup_t] = -1e9
rows, md = [], ["# Competitor finder: worked examples (test companies, LSA)\n"]
for i in rng.choice(np.where(valid_t)[0], 30, replace=False):
    top = np.argsort(-S[i])[:5]; md_i = [f"## {te.iloc[i]['name']}  ({te.iloc[i]['label']})", f"> {texts[i][:200]}", ""]
    for j in top:
        w, base, drop = occlusion("LSA", texts[i], CV["LSA"][j]); why = [w[k].strip(".,;:()") for k in np.argsort(-drop)[:3]]
        shared = sorted(set(te.iloc[i]["taglist"]) & set(cor.iloc[j]["taglist"]))
        rows.append({"query": te.iloc[i]["name"], "competitor": cor.iloc[j]["name"], "competitor_industry": cor.iloc[j]["label"], "similarity": round(float(S[i, j]), 3),
                     "shared_tags": "|".join(shared), "why_words": " ".join(why), "rating_1to5": None})
        md_i.append(f"- **{cor.iloc[j]['name']}** ({cor.iloc[j]['label']}) sim {S[i, j]:.2f}; shared tags: {', '.join(shared) or 'none'}; matched on: {', '.join(why)}")
    md.append("\n".join(md_i) + "\n")
pd.DataFrame(rows).to_csv(f"{RES}/explanations_to_rate.csv", index=False); open(f"{RES}/worked_examples.md", "w").write("\n".join(md[:13]))

# ---- 2. faithfulness: delete the top-3 "why" words vs 3 random words; similarity to the top-1 competitor should fall more ----
def faith(enc_name, n=120):
    dr_top, dr_rand = [], []
    Qe = ENC[enc_name](texts); Sx = Qe @ CV[enc_name].T; Sx[dup_t] = -1e9
    for i in rng.choice(np.where(valid_t)[0], n, replace=False):
        j = int(np.argmax(Sx[i])); w, base, drop = occlusion(enc_name, texts[i], CV[enc_name][j])
        if len(w) < 8: continue
        top3 = set(np.argsort(-drop)[:3]); rnd = [set(rng.choice(len(w), 3, replace=False)) for _ in range(3)]
        sim = lambda ks: float(ENC[enc_name]([" ".join(x for k, x in enumerate(w) if k not in ks)])[0] @ CV[enc_name][j])
        dr_top.append(base - sim(top3)); dr_rand.append(np.mean([base - sim(r) for r in rnd]))
    return {"n": len(dr_top), "drop_top3_why_words": round(float(np.mean(dr_top)), 3), "drop_3_random_words": round(float(np.mean(dr_rand)), 3),
            "ratio": round(float(np.mean(dr_top) / max(np.mean(dr_rand), 1e-9)), 1)}
out["faithfulness"] = {k: faith(k) for k in ("LSA", "CNN")}; print(out["faithfulness"], flush=True)

# ---- 3. robustness: how much does the competitor list change when the description is perturbed? ----
def typo(t):
    ws = t.split()
    for j in rng.choice(len(ws), max(1, len(ws) // 10), replace=False):
        w = ws[j]
        if len(w) > 3: k = rng.integers(1, len(w) - 2); ws[j] = w[:k] + w[k + 1] + w[k] + w[k + 2:]
    return " ".join(ws)
sub = rng.choice(np.where(valid_t)[0], 300, replace=False)
def topk_of(T, idx):
    Sx = ENC["LSA"](T) @ CV["LSA"].T; Sx[dup_t[idx]] = -1e9; return np.argsort(-Sx, axis=1)[:, :10], Sx
base_top, _ = topk_of([texts[i] for i in sub], sub); out["robustness"] = {}
perts = {"typos (10% of words)": lambda t: typo(t), "first sentence only": lambda t: re.split(r"(?<=[.!?])\s", t)[0],
         "random 30% of words removed": lambda t: " ".join(w for w in t.split() if rng.random() > 0.3)}
for name, f in perts.items():
    top2, _ = topk_of([f(texts[i]) for i in sub], sub)
    ov = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(base_top, top2)])
    p10 = np.mean([rel_t[i, t].mean() for i, t in zip(sub, top2)])
    out["robustness"][name] = {"top10_overlap_with_original": round(float(ov), 3), "p10": round(float(p10), 3)}
out["robustness"]["unperturbed_p10"] = round(float(np.mean([rel_t[i, t].mean() for i, t in zip(sub, base_top)])), 3)

# ---- 4. reliability: by length, industry, era; confidence = top-1 similarity; abstain; off-topic text ----
ev = evaluate(S, rel_t, ind_t, valid_t, dup_t); p10 = ev["p10"]; v = np.where(valid_t)[0]; top1 = np.sort(S, axis=1)[:, -1][v]
wl = te["text_words"].values[v]; q = np.quantile(wl, [1 / 3, 2 / 3]); g = np.digitize(wl, q)
out["p10_by_length"] = {n: round(float(p10[g == k].mean()), 3) for k, n in enumerate(["short", "medium", "long"])}
lab = te["label"].values[v]; out["p10_by_industry"] = {c: {"p10": round(float(p10[lab == c].mean()), 3), "n": int((lab == c).sum())} for c in sorted(set(lab)) if (lab == c).sum() >= 15}
era = te["batch_year"].values[v] < 2025; out["p10_by_era"] = {"before_2025": round(float(p10[era].mean()), 3), "2025_on": round(float(p10[~era].mean()), 3)}
qs = np.quantile(top1, [0.2, 0.4, 0.6, 0.8]); b = np.digitize(top1, qs)
out["p10_by_top1_similarity_quintile"] = {f"Q{k+1}": round(float(p10[b == k].mean()), 3) for k in range(5)}
Sv = ENC["LSA"](va["text"].tolist()) @ ENC["LSA"](tr["text"].tolist()).T; thr = float(np.quantile(Sv.max(1), 0.2))      # threshold from VALIDATION only
keep = top1 >= thr; out["abstain"] = {"threshold_from_val_20th_pct": round(thr, 3), "coverage": round(float(keep.mean()), 3), "p10_on_kept": round(float(p10[keep].mean()), 3), "p10_on_abstained": round(float(p10[~keep].mean()), 3)}
ood = ["We sell second hand cars and run a local garage with six mechanics.", "Our bakery makes sourdough and pastries for cafes in the city centre.",
       "The poem explores memory and loss through the voice of a river.", "Weekly yoga classes and meditation retreats for busy parents."]
o_top1 = (ENC["LSA"](ood) @ CV["LSA"].T).max(1); out["off_topic_top1_similarity"] = {"texts": [round(float(x), 2) for x in o_top1], "in_domain_median": round(float(np.median(top1)), 2), "below_threshold": [bool(x < thr) for x in o_top1]}
t = time.time(); _ = ENC["LSA"]([texts[0]]) @ CV["LSA"].T; out["latency_ms_per_query_LSA"] = round((time.time() - t) * 1000, 1)
t = time.time(); _ = ENC["CNN"]([texts[0]]) @ CV["CNN"].T; out["latency_ms_per_query_CNN"] = round((time.time() - t) * 1000, 1)
np.save(f"{RES}/_p10_top1.npy", np.c_[p10, top1])
json.dump(out, open(f"{RES}/results_task8_competitors.json", "w"), indent=2, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "faithfulness"}, indent=1)); print("DONE")
