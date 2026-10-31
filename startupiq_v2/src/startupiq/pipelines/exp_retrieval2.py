"""Validation-only experiments, round 2 (hub correction, query expansion, one-liner embeddings, structured boosts).
Run: PYTHONPATH=src python -m startupiq.pipelines.exp_retrieval2"""
import warnings, json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from startupiq.pipelines.comp_common import prep, relevance, evaluate, masks
from startupiq.pipelines.embed_cache import path
from startupiq.pipelines.yc_split import load_split
warnings.filterwarnings("ignore")
D = prep(); tr, va = D["tr"], D["va"]; df, *_ = load_split(); pos = {ix: k for k, ix in enumerate(df.index)}
M = "BAAI/bge-small-en-v1.5"; Eb = np.load(path(M, 128)); Eo = np.load(path(M, 64, "one_liner_clean"))
take = lambda E, d: E[[pos[i] for i in d.index]]
rel = relevance(D["T_va"], D["T_tr"]); ind, dup = masks(va, tr); valid = D["T_va"].sum(1) > 0
z = lambda S: (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english").fit(tr.text)
W, Wv = tf.transform(tr.text), tf.transform(va.text); S_tf = (Wv @ W.T).toarray()
S_full = take(Eb, va) @ take(Eb, tr).T; S_one = take(Eo, va) @ take(Eo, tr).T
def sc(S, name):
    r = evaluate(S, rel, ind, valid, dup); print(f"{name:46s} p10={r['p10'].mean():.3f} sameInd={r['ind10'].mean():.3f}", flush=True); return r["p10"].mean()
sc(S_full, "bge full text"); sc(S_one, "bge one-liner"); sc(z(S_full) + z(S_one), "bge full + one-liner")
base = 0.7 * z(S_full) + 0.3 * z(S_tf); sc(base, "full+tfidf (0.7/0.3)")
b2 = 0.4 * z(S_full) + 0.3 * z(S_one) + 0.3 * z(S_tf); sc(b2, "full+oneliner+tfidf")
b3 = 0.35 * z(S_full) + 0.35 * z(S_one) + 0.3 * z(S_tf) + 1.0 * ind; sc(b3, "... + industry boost 1.0")
# hub correction (CSLS): penalise corpus items that are near everything
Ctr = take(Eb, tr); Str = Ctr @ Ctr.T; np.fill_diagonal(Str, -1); k = 10
r_c = np.sort(S_full, axis=0)[-k:].mean(0)        # mean sim of each corpus item to its k nearest val queries
for lam in (0.5, 1.0): sc(base - lam * z(r_c[None, :].repeat(len(va), 0)) * 0.3, f"base - hub penalty {lam}")
r_c2 = np.sort(Str, axis=1)[:, -k:].mean(1)      # corpus-internal hubness (no query info)
for lam in (0.3, 0.6, 1.0): sc(b3 - lam * (r_c2 - r_c2.mean()) / r_c2.std(), f"b3 - corpus hub penalty {lam}")
# pseudo-relevance feedback: q' = q + a * mean(top-m neighbours); rescore
for m_, a in ((5, 0.5), (5, 1.0), (10, 1.0)):
    S0 = b3.copy(); S0[dup] = -1e9; top = np.argsort(-S0, 1)[:, :m_]
    Ctf = (W.T).tocsr()
    Sf = np.stack([Str_row for Str_row in [(Ctr[t].mean(0) @ Ctr.T) for t in top]])
    sc(b3 + a * z(Sf), f"b3 + PRF(m={m_}, a={a})")
