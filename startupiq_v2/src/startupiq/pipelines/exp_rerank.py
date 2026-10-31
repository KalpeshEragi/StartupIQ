"""Cross-encoder rerank test on a validation subset: does reranking the top-30 of the fused retriever improve precision@10?
Run: PYTHONPATH=src python -m startupiq.pipelines.exp_rerank CROSS_ENCODER_NAME NQ"""
import sys, time, warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from startupiq.pipelines.comp_common import prep, relevance, masks
from startupiq.pipelines.embed_cache import path
from startupiq.pipelines.yc_split import load_split
warnings.filterwarnings("ignore")
name, nq = sys.argv[1], int(sys.argv[2]); T0 = time.time()
D = prep(); tr, va = D["tr"], D["va"]; df, *_ = load_split(); pos = {ix: k for k, ix in enumerate(df.index)}
M = "BAAI/bge-small-en-v1.5"; Eb, Eo = np.load(path(M, 128)), np.load(path(M, 64, "one_liner_clean")); take = lambda E, d: E[[pos[i] for i in d.index]]
rel = relevance(D["T_va"], D["T_tr"]); ind, dup = masks(va, tr); valid = D["T_va"].sum(1) > 0
z = lambda S: (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english").fit(tr.text)
S = 0.35 * z(take(Eb, va) @ take(Eb, tr).T) + 0.35 * z(take(Eo, va) @ take(Eo, tr).T) + 0.3 * z((tf.transform(va.text) @ tf.transform(tr.text).T).toarray()) + 1.0 * ind
S[dup] = -1e9
rng = np.random.default_rng(1); qs = rng.choice(np.where(valid)[0], nq, replace=False)
from sentence_transformers import CrossEncoder
ce = CrossEncoder(name, device="cpu", max_length=128)
txt = lambda d, i: str(d.iloc[i]["text"])[:350]
p_base, p_rr, p_mix = [], [], []
for qi in qs:
    if time.time() - T0 > 150: break
    top = np.argsort(-S[qi])[:30]; sc = ce.predict([(txt(va, qi), txt(tr, c)) for c in top], batch_size=32, show_progress_bar=False)
    sc = np.asarray(sc, float); fused = S[qi][top]
    p_base.append(rel[qi, top[:10]].mean()); p_rr.append(rel[qi, top[np.argsort(-sc)[:10]]].mean())
    p_mix.append(rel[qi, top[np.argsort(-(z(fused[None])[0] + z(sc[None])[0]))[:10]]].mean())
print(name, "queries", len(p_base), "base", round(np.mean(p_base), 3), "rerank only", round(np.mean(p_rr), 3), "fused+rerank", round(np.mean(p_mix), 3), "secs", round(time.time() - T0))
