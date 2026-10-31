"""Business-type classifier: is this a clothing shop, an electronics shop, a kirana, a restaurant, a pharmacy or a stationer?

python -m startupiq.category     train six classifiers on the SYNTHETIC businesses (synthetic.py), choose on
                                 validation, score the test shops once, save data/models/category.joblib

features() is the one function that turns a sales register into numbers, used for training and for every
upload, so the model never sees a feature computed differently from how it was trained. It works with what
the shop has: date and sales are enough; orders (bills), items sold and returns add the ticket features;
daily rows add the weekday features. Missing pieces become NaN and the models are trained with them missing.
"""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from startupiq.synthetic import DIWALI, NAMES

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "data" / "models" / "category.joblib"
SURE = 0.5  # below this top probability the card says "not sure"

TICKET = ["log_avg_bill", "items_per_bill", "log_item_price", "return_rate"]
DAILY = ["weekend_share", "sunday_share", "month_start_share"]
SEASON = ["diwali_lift"] + [f"m{m:02d}" for m in range(1, 13)]
SHAPE = ["volatility", "zero_share", "trend"]
FEATURES = TICKET + DAILY + SEASON + SHAPE
LABELS = {"log_avg_bill": "Average bill (log10 ₹)", "items_per_bill": "Items per bill", "log_item_price": "Price per item (log10 ₹)",
          "return_rate": "Returns as share of sales", "weekend_share": "Share of sales on weekends", "sunday_share": "Share on Sundays",
          "month_start_share": "Share in the first week of the month", "diwali_lift": "Lift in the weeks before Diwali",
          "volatility": "Week-to-week swing", "zero_share": "Weeks with no sales", "trend": "Yearly growth (log)",
          **{f"m{m:02d}": f"{pd.Timestamp(2024, m, 1):%B} vs a normal month" for m in range(1, 13)}}


def features(df: pd.DataFrame) -> dict[str, float]:
    """df: one row per day or per week with 'date' and 'sales', optionally 'orders', 'items', 'returns'."""
    df = df.sort_values("date")
    d = pd.to_datetime(df["date"])
    sales = df["sales"].astype(float).clip(lower=0)
    total = sales.sum()
    f = dict.fromkeys(FEATURES, np.nan)
    if total <= 0:
        return f
    orders = df["orders"].astype(float).sum() if "orders" in df else 0
    items = df["items"].astype(float).sum() if "items" in df else 0
    if orders > 0:
        f["log_avg_bill"] = np.log10(total / orders)
        if items > 0:
            f["items_per_bill"] = items / orders
    if items > 0:
        f["log_item_price"] = np.log10(total / items)
    if "returns" in df:
        f["return_rate"] = df["returns"].astype(float).sum() / total
    daily = len(d) > 1 and d.diff().dt.days.median() <= 1.5
    if daily:
        f["weekend_share"] = sales[d.dt.dayofweek >= 5].sum() / total
        f["sunday_share"] = sales[d.dt.dayofweek == 6].sum() / total
        f["month_start_share"] = sales[d.dt.day <= 7].sum() / total
    weekly = pd.Series(sales.values, index=d).resample("W-SUN").sum()
    if len(weekly) >= 8:
        y = weekly.to_numpy()
        x = np.arange(len(y))
        fit = np.polyfit(x, np.log1p(y), 1)
        ratio = (y + 1) / np.exp(np.polyval(fit, x))  # weekly sales over the shop's own trend line
        months = weekly.index.month
        for m in range(1, 13):
            if (months == m).sum() >= 2:
                f[f"m{m:02d}"] = float(ratio[months == m].mean() - 1)
        near = np.zeros(len(y), bool)
        for dw in DIWALI:
            near |= (weekly.index >= dw - pd.Timedelta(days=14)) & (weekly.index <= dw + pd.Timedelta(days=8))
        if near.any() and (~near).sum() >= 4:
            f["diwali_lift"] = float(ratio[near].mean() / ratio[~near].mean() - 1)
        f["volatility"] = float(np.std(np.diff(np.log1p(y))))
        f["zero_share"] = float(np.mean(y == 0))
        f["trend"] = float(fit[0] * 52)
    return f


def _matrix(rows: list[dict]) -> np.ndarray:
    return np.array([[r[k] for k in FEATURES] for r in rows], float)


def models(seed: int = 0) -> dict:
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    def prep():  # median fill plus a 0/1 flag per feature that was missing, then scale
        return [SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler()]
    return {
        "Logistic regression": make_pipeline(*prep(), LogisticRegression(max_iter=2000, C=1.0)),
        "k-nearest neighbours (k=15)": make_pipeline(*prep(), KNeighborsClassifier(15, weights="distance")),
        "SVM (RBF kernel)": make_pipeline(*prep(), SVC(C=3.0, probability=True, random_state=seed)),
        "Random forest": make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                                       RandomForestClassifier(500, min_samples_leaf=2, n_jobs=-1, random_state=seed)),
        "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, random_state=seed, verbose=-1),
        "Neural network (MLP)": make_pipeline(*prep(), MLPClassifier((64, 32), alpha=1e-3, max_iter=600, early_stopping=True, random_state=seed)),
    }


def dataset(seed: int = 1) -> pd.DataFrame:
    """One row per synthetic business: features of an upload-like view of its register (some sales-only, some weekly)."""
    from startupiq.synthetic import OUT, upload_view

    meta = pd.read_parquet(OUT / "businesses.parquet")
    daily = pd.read_parquet(OUT / "daily.parquet")
    rng = np.random.default_rng(seed)
    rows = []
    for b, g in daily.groupby("business", sort=True):
        view, kind = upload_view(g.drop(columns="business"), rng)
        rows.append({"business": b, "view": kind, **features(view)})
    return pd.DataFrame(rows).merge(meta[["business", "category", "split", "weeks", "atypical"]], on="business")


def run() -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

    from startupiq.evaluate import report

    t0 = time.time()
    data = dataset()
    y = data.category.map({c: i for i, c in enumerate(NAMES)}).to_numpy()
    X = data[FEATURES].to_numpy(float)
    tr, va, te = ((data.split == s).to_numpy() for s in ("train", "val", "test"))
    rows = []
    for name, m in models().items():
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])
        rows.append({"model": name, "val accuracy": accuracy_score(y[va], p.argmax(1)), "val macro-F1": f1_score(y[va], p.argmax(1), average="macro"),
                     "val log-loss": log_loss(y[va], p, labels=range(len(NAMES)))})
    majority = np.bincount(y[tr]).argmax()
    rows.append({"model": "Majority class (baseline)", "val accuracy": accuracy_score(y[va], np.full(va.sum(), majority)),
                 "val macro-F1": f1_score(y[va], np.full(va.sum(), majority), average="macro"), "val log-loss": np.nan})
    val = pd.DataFrame(rows)
    chosen = val.iloc[:-1].sort_values(["val macro-F1", "val log-loss"], ascending=[False, True]).iloc[0]["model"]

    test_rows = []  # every model refit on train + validation, test shops scored once
    fitted = {}
    for name, m in models().items():
        m.fit(X[tr | va], y[tr | va])
        fitted[name] = m
        p = m.predict_proba(X[te])
        test_rows.append({"model": name, "test accuracy": accuracy_score(y[te], p.argmax(1)),
                          "test macro-F1": f1_score(y[te], p.argmax(1), average="macro"), "test log-loss": log_loss(y[te], p, labels=range(len(NAMES)))})
    test = pd.DataFrame(test_rows)
    best = fitted[chosen]
    p = best.predict_proba(X[te])
    pred = p.argmax(1)
    d = data[te].assign(pred=pred, ok=pred == y[te], conf=p.max(1))
    by_view = d.groupby("view").ok.agg(["mean", "size"]).rename(columns={"mean": "accuracy", "size": "shops"}).reset_index()
    d["history"] = pd.cut(d.weeks, [0, 26, 52, 200], labels=["12-26 weeks", "27-52 weeks", "53+ weeks"])
    by_hist = d.groupby("history", observed=True).ok.agg(["mean", "size"]).rename(columns={"mean": "accuracy", "size": "shops"}).reset_index()
    by_type = d.groupby("atypical").ok.mean().rename({False: "typical shops", True: "atypical shops"}).reset_index()
    sure = d[d.conf >= SURE]
    cm = pd.DataFrame(confusion_matrix(y[te], pred), index=[f"true {c}" for c in NAMES], columns=[f"pred {c.split(' ')[0]}" for c in NAMES])
    per_class = pd.DataFrame({"category": NAMES, "F1": f1_score(y[te], pred, average=None)})
    imp = None
    if chosen in ("LightGBM", "Random forest"):
        est = best if chosen == "LightGBM" else best[-1]
        vals = est.feature_importances_[:len(FEATURES)]
        imp = pd.DataFrame({"feature": [LABELS[f] for f in FEATURES], "importance": vals / vals.sum()}).sort_values("importance", ascending=False).head(10)

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best, "name": chosen, "features": FEATURES, "classes": NAMES,
                 "test_accuracy": float((pred == y[te]).mean()), "medians": data[tr | va].groupby("category")[FEATURES].median()}, MODEL)
    out = {"validation": val.round(4).to_dict("records"), "chosen": chosen, "test": test.round(4).to_dict("records"),
           "by_view": by_view.round(4).to_dict("records"), "by_history": by_hist.round(4).to_dict("records"),
           "confident_share": round(len(sure) / len(d), 4), "confident_accuracy": round(float(sure.ok.mean()), 4),
           "seconds": round(time.time() - t0)}
    sections = [
        ("Data", f"SYNTHETIC: {len(data)} businesses, 6 categories, split by business (train {tr.sum()}, validation {va.sum()}, test {te.sum()}). "
                 "Each shop is seen the way it might be uploaded: 35% date and sales only, 15% without returns, 25% weekly totals. "
                 f"{len(FEATURES)} features: {', '.join(FEATURES)}."),
        ("Validation (choose here)", report.table(val.round(4))),
        ("Chosen", f"**{chosen}**: highest validation macro-F1 (ties broken by log-loss)."),
        ("Test (refit on train + validation, scored once)", report.table(test.round(4))),
        ("Chosen model on test, by what the upload contains", report.table(by_view.round(4))),
        ("By length of history", report.table(by_hist.round(4))),
        ("Typical vs atypical shops", report.table(by_type.round(4))),
        ("Confidence", f"{len(sure) / len(d):.0%} of test shops get a top probability of {SURE:.0%} or more; accuracy on those: {sure.ok.mean():.1%}. "
                       "The rest are shown as 'not sure'."),
        ("Per category F1 (test)", report.table(per_class.round(4))),
        ("Confusion matrix (test)", report.table(cm.reset_index().rename(columns={"index": ""}))),
    ] + ([("Most used features", report.table(imp.round(4)))] if imp is not None else [])
    report.write("category", "Business-type classifier (SYNTHETIC data)", sections, out, command="python -m startupiq.category")
    return out


@lru_cache(maxsize=1)
def load() -> dict | None:
    return joblib.load(MODEL) if MODEL.exists() else None


def classify(df: pd.DataFrame) -> dict | None:
    """Category probabilities for an upload, with the features that most separate the top two categories."""
    m = load()
    if m is None or df is None or len(df) < 8:
        return None
    f = features(df)
    p = m["model"].predict_proba(np.array([[f[k] for k in FEATURES]], float))[0]
    order = np.argsort(p)[::-1]
    top, second = m["classes"][order[0]], m["classes"][order[1]]
    med = m["medians"]
    evidence = []
    for k in FEATURES:  # features you have, ranked by how far apart the top two categories usually are on them
        if np.isnan(f[k]):
            continue
        a, b = med.loc[top, k], med.loc[second, k]
        spread = med[k].std() + 1e-9
        evidence.append({"key": k, "label": LABELS[k], "yours": round(float(f[k]), 3), "typical": round(float(a), 3),
                         "runner_up": round(float(b), 3), "_gap": abs(a - b) / spread})
    evidence = sorted(evidence, key=lambda e: -e["_gap"])[:5]
    for e in evidence:
        e.pop("_gap")
    have = [n for n, cols in (("orders and items", TICKET[:3]), ("returns", TICKET[3:]), ("daily rows", DAILY)) if not np.isnan(f[cols[0]])]
    return {"category": top if p[order[0]] >= SURE else None, "best_guess": top, "confidence": round(float(p[order[0]]), 3),
            "probabilities": [{"category": m["classes"][i], "p": round(float(p[i]), 3)} for i in order],
            "runner_up": second, "evidence": evidence, "inputs": have, "model": m["name"], "test_accuracy": round(m["test_accuracy"], 3),
            "vector": [float(x) for x in p]}


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=1, default=str)[:3000])
