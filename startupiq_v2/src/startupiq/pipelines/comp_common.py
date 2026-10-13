"""Shared pieces for the competitor finder (Tasks 7 and 8): data, tag relevance, retrieval metrics."""
import warnings
from collections import Counter
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from startupiq.pipelines.yc_split import load_split
from startupiq.models.textnn import embed

N_TAGS = 50
warnings.filterwarnings('ignore', message='unknown class')


def parse_tags(s):
    return [t.strip() for t in str(s).split("|") if t.strip()] if isinstance(s, str) else []


def prep():
    df, tr, va, te = load_split()
    tr, va, te = tr.copy(), va.copy(), te.copy()
    industry = set(df["industry"].unique()) | {"B2B", "Consumer"}          # tags that just restate the label are dropped
    for d in (tr, va, te):
        d["taglist"] = [[t for t in parse_tags(s) if t not in industry] for s in d["tags"]]
    cnt = Counter(t for tl in tr["taglist"] for t in tl)
    top = [t for t, _ in cnt.most_common(N_TAGS)]
    mlb_all = MultiLabelBinarizer(sparse_output=False).fit(list(tr["taglist"]) + list(va["taglist"]) + list(te["taglist"]))
    mlb_top = MultiLabelBinarizer(classes=top)
    out = {"df": df, "tr": tr, "va": va, "te": te, "top_tags": top}
    for k, d in (("tr", tr), ("va", va), ("te", te)):
        out["T_" + k] = mlb_all.transform(list(d["taglist"])).astype(np.float32)      # all non-industry tags
        out["Y_" + k] = mlb_top.fit_transform(list(d["taglist"])).astype(np.float32)   # top-50 targets
    return out


def relevance(Tq, Tc):
    """rel[q, c] = the two companies share at least one non-industry tag."""
    return (Tq @ Tc.T) > 0


def cnn_vectors(model, X, mu=None):
    E = embed(model, X)
    if mu is None: mu = E.mean(0)
    return normalize(E - mu), mu


def evaluate(S, rel, same_ind, valid, dup, k=10):
    """Per-query precision@10 (neighbour shares a tag), precision@1, same-industry@10, for queries that have tags."""
    S = S.copy(); S[dup] = -1e9
    top = np.argsort(-S, axis=1)[:, :k]; r = np.arange(len(S))[:, None]
    p10 = rel[r, top].mean(1); p1 = rel[np.arange(len(S)), top[:, 0]].astype(float); ind = same_ind[r, top].mean(1)
    return {"p10": p10[valid], "p1": p1[valid], "ind10": ind[valid], "top": top}


def boot_ci(x, n=1000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [round(float(v), 3) for v in np.percentile(m, [2.5, 97.5])]


def paired_ci(a, b, n=1000, seed=0):
    return boot_ci(np.asarray(a) - np.asarray(b), n, seed)


def masks(dq, dc):
    """Same-industry matrix and near-duplicate mask (a company's copy in the corpus must not count as its own competitor)."""
    ind = dq["label"].values[:, None] == dc["label"].values[None, :]
    dup = (dq["dup_group"].values[:, None] == dc["dup_group"].values[None, :]) & dq["dup_group"].notna().values[:, None]
    same_dom = (dq["domain"].values[:, None] == dc["domain"].values[None, :]) & dq["domain"].notna().values[:, None]
    return ind, dup | same_dom
