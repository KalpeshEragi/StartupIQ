"""Tasks 5-6 on Stream B (Form D): funding-amount regression, amount bands, quantile range.
Split grouped by issuer (CIK). amount_sold is excluded (it is the same quantity as the target)."""
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

SEED = 42; RES = "docs/results/task5_6"; FIG = f"{RES}/figures"; BLUE, GRAY, INK = "#2a78d6", "#b8bcc2", "#222222"
os.makedirs(FIG, exist_ok=True); out = {}
df = pd.read_csv("data/processed/formd_clean.csv")
df = df[df["startup_like_group"] & df["log10_offering"].notna() & df["cik"].notna()].reset_index(drop=True)
out["n_rows"] = len(df); out["n_issuers"] = int(df["cik"].nunique())
CAT = ["industry_group", "revenue_range", "entity_type", "state_or_country"]
NUM = ["issuer_age", "n_related_persons", "filing_quarter", "issuer_filing_count", "issuer_n_amendments"]
BOOL = ["is_equity", "is_debt", "incorporated_within_5y"]
out["features"] = CAT + NUM + BOOL
out["excluded"] = {"amount_sold, n_investors, min_investment": "same event / post-outcome", "log10_offering*": "target"}
df["state_or_country"] = df["state_or_country"].fillna("NA"); df["revenue_range"] = df["revenue_range"].fillna("blank")
for b in BOOL: df[b] = df[b].astype(float)
y = df["log10_offering"].values
g = df["cik"].values
tv, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(df, y, g))
tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(df.iloc[tv], y[tv], g[tv]))
tr, va = tv[tr], tv[va]
out["split"] = {"train": len(tr), "val": len(va), "test": len(te)}
assert not set(g[tr]) & set(g[te]) and not set(g[tr]) & set(g[va])


def pre(scale=False):
    num = [("imp", SimpleImputer(strategy="median", add_indicator=True))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), CAT),
                              ("n", Pipeline(num), NUM), ("b", "passthrough", BOOL)])


def ev(m, idx):
    p = m.predict(df.iloc[idx]); t = y[idx]
    return {"MAE_log10": round(mean_absolute_error(t, p), 3), "RMSE_log10": round(mean_squared_error(t, p) ** .5, 3),
            "R2": round(r2_score(t, p), 3), "median_APE": round(float(np.median(np.abs(10 ** p - 10 ** t) / 10 ** t)), 3)}


# segment-median baseline (industry_group x revenue_range)
seg = df.iloc[tr].groupby(["industry_group", "revenue_range"])["log10_offering"].median()
glob = np.median(y[tr])
class SegMed:
    def predict(self, d): return np.array([seg.get((a, b), glob) for a, b in zip(d["industry_group"], d["revenue_range"])])
reg = {"segment-median baseline": SegMed(), "global median": DummyRegressor(strategy="median").fit(df.iloc[tr], y[tr])}
mods = {"linear regression": LinearRegression(), "ridge": Ridge(10), "random forest": RandomForestRegressor(300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        "gradient boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=.8, random_state=SEED)}
for n, m in mods.items():
    reg[n] = Pipeline([("p", pre(scale=n in ("linear regression", "ridge"))), ("m", m)]).fit(df.iloc[tr], y[tr])
out["regression_val"] = {n: ev(m, va) for n, m in reg.items()}
bestr = min((k for k in mods), key=lambda k: out["regression_val"][k]["MAE_log10"]); out["regression_chosen"] = bestr
out["regression_test"] = {bestr: ev(reg[bestr], te), "segment-median baseline": ev(reg["segment-median baseline"], te)}

# bands: tertiles from train only
q = np.quantile(y[tr], [1 / 3, 2 / 3]); band = lambda v: np.digitize(v, q)
out["band_edges_usd"] = [int(10 ** x) for x in q]
cls = {"majority": DummyClassifier(strategy="most_frequent"), "logistic regression": LogisticRegression(max_iter=2000),
       "decision tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, random_state=SEED),
       "random forest": RandomForestClassifier(300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
       "hist gradient boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, random_state=SEED)}
cf = {}; cv = {}
for n, m in cls.items():
    sc = n in ("logistic regression",)
    cf[n] = Pipeline([("p", pre(scale=sc)), ("m", m)]).fit(df.iloc[tr], band(y[tr]))
    p = cf[n].predict(df.iloc[va]); cv[n] = {"acc": round(accuracy_score(band(y[va]), p), 3), "macro_f1": round(f1_score(band(y[va]), p, average="macro"), 3)}
out["band_val"] = cv
bestc = max((k for k in cls if k != "majority"), key=lambda k: cv[k]["macro_f1"]); out["band_chosen"] = bestc
p = cf[bestc].predict(df.iloc[te]); out["band_test"] = {"model": bestc, "acc": round(accuracy_score(band(y[te]), p), 3), "macro_f1": round(f1_score(band(y[te]), p, average="macro"), 3)}
out["band_test_majority"] = round(accuracy_score(band(y[te]), cf["majority"].predict(df.iloc[te])), 3)
# within-one-band accuracy (adjacent bands count as close)
out["band_test_within_one"] = round(float((np.abs(p - band(y[te])) <= 1).mean()), 3)

# quantile range 10/50/90
qm = {a: Pipeline([("p", pre()), ("m", GradientBoostingRegressor(loss="quantile", alpha=a, n_estimators=200, max_depth=3, learning_rate=.05, random_state=SEED))]).fit(df.iloc[tr], y[tr]) for a in (.1, .5, .9)}
lo, md, hi = (qm[a].predict(df.iloc[te]) for a in (.1, .5, .9))
out["quantile_range_test"] = {"coverage_10_90": round(float(((y[te] >= lo) & (y[te] <= hi)).mean()), 3), "nominal": 0.8,
                              "median_width_log10": round(float(np.median(hi - lo)), 2)}

# ----- figures
def sty(ax, t):
    ax.set_title(t, loc="left", fontsize=12, color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
names = ["global median", "segment-median baseline"] + list(mods); v = [out["regression_val"][n]["MAE_log10"] for n in names]
axs[0].barh(names[::-1], v[::-1], color=[BLUE if n == bestr else GRAY for n in names][::-1])
for i, x in enumerate(v[::-1]): axs[0].text(x, i, f" {x:.3f}", va="center")
sty(axs[0], "Regression: validation MAE on log10 amount (lower is better)")
pt = reg[bestr].predict(df.iloc[te]); axs[1].scatter(y[te], pt, s=4, alpha=.3, color=BLUE); axs[1].plot([3, 9], [3, 9], color=INK, lw=1)
axs[1].set_xlabel("actual log10 offering"); axs[1].set_ylabel("predicted"); axs[1].set_xlim(3, 9); axs[1].set_ylim(3, 9)
sty(axs[1], f"Test: {bestr} (R2 {out['regression_test'][bestr]['R2']})"); fig.tight_layout(); fig.savefig(f"{FIG}/t6_formd_regression.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 4.5)); names = list(cv)
ax.barh(names[::-1], [cv[n]["acc"] for n in names][::-1], color=[BLUE if n == bestc else GRAY for n in names][::-1])
for i, n in enumerate(names[::-1]): ax.text(cv[n]["acc"], i, f" acc {cv[n]['acc']:.3f}, F1 {cv[n]['macro_f1']:.3f}", va="center", fontsize=9)
ax.set_xlim(0, 1); sty(ax, "Funding-amount band (3 classes), validation"); fig.tight_layout(); fig.savefig(f"{FIG}/t6_formd_bands.png", dpi=150); plt.close(fig)
json.dump(out, open(f"{RES}/results_formd.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in out if k not in ("features", "excluded")}, indent=1))
