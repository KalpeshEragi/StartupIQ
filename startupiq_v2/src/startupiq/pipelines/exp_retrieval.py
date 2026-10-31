"""Validation-only retrieval experiments (corpus = train, queries = val). Test is NOT touched here.
Run: PYTHONPATH=src python -m startupiq.pipelines.exp_retrieval MODEL [MAXLEN]"""
import sys, warnings, json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from startupiq.pipelines.comp_common import prep, relevance, evaluate, masks
from startupiq.pipelines.embed_cache import path
warnings.filterwarnings("ignore")
D = prep(); tr, va = D["tr"], D["va"]
model = sys.argv[1]; ml = int(sys.argv[2]) if len(sys.argv) > 2 else 256
from startupiq.pipelines.yc_split import load_split
df, *_ = load_split(); E = np.load(path(model, ml)); pos = {ix: k for k, ix in enumerate(df.index)}
Etr, Eva = E[[pos[i] for i in tr.index]], E[[pos[i] for i in va.index]]
rel = relevance(D["T_va"], D["T_tr"]); ind, dup = masks(va, tr); valid = D["T_va"].sum(1) > 0
specific_gen = {"SaaS", "Marketplace", "Artificial Intelligence", "AI", "Analytics", "Productivity", "Developer Tools", "E-commerce", "Social", "Media"}
def score(S, name):
    r = evaluate(S, rel, ind, valid, dup); print(f"{name:34s} p10={r['p10'].mean():.3f} p1={r['p1'].mean():.3f} sameInd@10={r['ind10'].mean():.3f}", flush=True); return r
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english").fit(tr.text)
W = tf.transform(tr.text); Wv = tf.transform(va.text); svd = TruncatedSVD(200, random_state=42).fit(W)
z = lambda S: (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
S_tf = (Wv @ W.T).toarray(); S_lsa = normalize(svd.transform(Wv)) @ normalize(svd.transform(W)).T; S_emb = Eva @ Etr.T
score(S_tf, "TFIDF"); score(S_lsa, "LSA"); score(S_emb, model.split("/")[-1])
for w in (0.3, 0.5, 0.7): score(w * z(S_emb) + (1 - w) * z(S_tf), f"hybrid emb/tfidf w={w}")
# structured feature: same industry boost
for b in (0.5, 1.0): score(z(S_emb) + b * ind, f"emb + industry boost {b}")
