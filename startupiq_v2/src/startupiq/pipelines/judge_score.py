"""Join blind judgements with system identities and score judged precision@10 per system.
Run: PYTHONPATH=src python -m startupiq.pipelines.judge_score val"""
import sys, json
import numpy as np, pandas as pd
from startupiq.pipelines.comp_common import boot_ci
split = sys.argv[1]; L = json.load(open(f"docs/results/judging/{split}_labels_claude.json"))["relevant"]
df = pd.read_csv(f"data/interim/judge/{split}_pairs_with_systems.csv"); df["k"] = df.groupby("qid", sort=False).cumcount()
df["rel"] = [int(k in L[str(q)]) for q, k in zip(df.qid, df.k)]
res = {}
for s in sorted({x for v in df.systems for x in v.split("|")}):
    m = df[df.systems.str.contains(s)]; per_q = m.groupby("qid").rel.sum() / 10
    res[s] = {"judged_p10": round(float(per_q.mean()), 3), "ci": boot_ci(per_q.values), "n_queries": int(len(per_q))}
print(json.dumps(res, indent=1)); json.dump(res, open(f"docs/results/judging/{split}_judged_precision.json", "w"), indent=2)
print("overall pooled relevance rate", round(float(df.rel.mean()), 3))
