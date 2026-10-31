"""Build a blind judging sheet: pooled top-10 of several systems for sampled queries, shuffled, system identity hidden.
Run: PYTHONPATH=src python -m startupiq.pipelines.judge_sheet val 20"""
import sys, json, warnings
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from startupiq.pipelines.comp_common import prep, masks
from startupiq.pipelines.embed_cache import path
from startupiq.pipelines.yc_split import load_split
warnings.filterwarnings("ignore")
split, nq = sys.argv[1], int(sys.argv[2]); rng = np.random.default_rng(7)
D = prep(); tr, va, te = D["tr"], D["va"], D["te"]
if split == "val": Q, C, TQ = va, tr, D["T_va"]
else: Q, C, TQ = te, pd.concat([tr, va]), D["T_te"]
df, *_ = load_split(); pos = {ix: k for k, ix in enumerate(df.index)}
E = np.load(path("BAAI/bge-small-en-v1.5", 128)); Eq, Ec = E[[pos[i] for i in Q.index]], E[[pos[i] for i in C.index]]
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english").fit(tr.text)
svd = TruncatedSVD(200, random_state=42).fit(tf.transform(tr.text)); Wq, Wc = tf.transform(Q.text), tf.transform(C.text)
z = lambda S: (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
ind, dup = masks(Q, C)
S = {"LSA": normalize(svd.transform(Wq)) @ normalize(svd.transform(Wc)).T, "BGE": Eq @ Ec.T}
S["HYB"] = 0.7 * z(S["BGE"]) + 0.3 * z((Wq @ Wc.T).toarray()) + 1.0 * ind
valid = np.where(TQ.sum(1) > 0)[0]; qs = rng.choice(valid, nq, replace=False); rows = []
for qi in qs:
    pool = {}
    for name, M in S.items():
        s = M[qi].copy(); s[dup[qi]] = -1e9
        for r, ci in enumerate(np.argsort(-s)[:10]): pool.setdefault(int(ci), []).append(name)
    for ci, systems in pool.items():
        rows.append({"qid": int(qi), "query": Q.iloc[qi]["name"], "q_text": Q.iloc[qi]["one_liner_clean"], "cand": C.iloc[ci]["name"], "c_text": C.iloc[ci]["one_liner_clean"], "systems": "|".join(systems)})
df_out = pd.DataFrame(rows).sample(frac=1, random_state=3).sort_values("qid", kind="stable").reset_index(drop=True)
df_out.to_csv(f"data/interim/judge/{split}_pairs_with_systems.csv", index=False)
with open(f"data/interim/judge/{split}_blind.txt", "w") as f:
    for q, g in df_out.groupby("qid", sort=False):
        f.write(f"\n### Q{q} [{g.iloc[0]['query']}] {str(g.iloc[0]['q_text'])[:140]}\n")
        for k, r in g.reset_index().iterrows(): f.write(f"{q}.{k} [{r['cand']}] {str(r['c_text'])[:110]}\n")
print(len(df_out), "pairs")
