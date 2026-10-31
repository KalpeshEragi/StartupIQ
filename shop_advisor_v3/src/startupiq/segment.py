"""Business profiling: which kind of seller is this? (the ML stage that runs before the forecast)

A sales history is reduced to seven unit-free behaviour features (how volatile, how intermittent,
how seasonal, how fast it is growing). K-means on 2,700 public weekly series (Rossmann stores and
UCI products) finds the segments offline; an upload is assigned to the nearest one, with a
confidence and an "unusual business" flag. Each segment carries a benchmark of the forecasting
methods on its own members, so the forecast card can say how methods behave for businesses like this one.

python -m startupiq.segment     fit, benchmark on the validation origin, write data/models/segments.joblib
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np

MODEL = Path(__file__).resolve().parents[2] / "data/models/segments.joblib"  # repo root, whatever the working folder
WINDOW = 52
FEATURES = ["volatility", "intermittency", "peak_ratio", "season_concentration", "trend", "momentum", "change_rate"]
LABELS = {
    "volatility": "Volatility (week-to-week spread)",
    "intermittency": "Weeks with no sales",
    "peak_ratio": "Peak vs a normal week",
    "season_concentration": "Share of sales in the best 8 weeks",
    "trend": "Half-year trend",
    "momentum": "Memory (this week predicts next)",
    "change_rate": "Typical weekly swing",
}
WINSOR = (0.01, 0.99)
K_RANGE = range(3, 7)


def profile(y) -> dict[str, float]:
    """Seven scale-free features from the last 52 weeks of a weekly series."""
    y = np.clip(np.asarray(y, float)[-WINDOW:], 0, None)
    if len(y) < WINDOW:
        raise ValueError(f"need {WINDOW} weeks, got {len(y)}")
    mean = y.mean() + 1e-9
    four = np.convolve(y, np.ones(4) / 4, mode="valid")
    acf1 = float(np.corrcoef(y[:-1], y[1:])[0, 1]) if y[:-1].std() > 0 and y[1:].std() > 0 else 0.0
    return {
        "volatility": float(y.std() / mean),
        "intermittency": float(np.mean(y == 0)),
        "peak_ratio": float(four.max() / mean),
        "season_concentration": float(np.sort(y)[-8:].sum() / (y.sum() + 1e-9)),
        "trend": float(np.log((y[26:].mean() + 1) / (y[:26].mean() + 1)) if mean > 0 else 0.0),
        "momentum": 0.0 if np.isnan(acf1) else acf1,
        "change_rate": float(np.abs(np.diff(y)).mean() / mean),
    }


def _vector(p: dict[str, float]) -> np.ndarray:
    return np.array([p[f] for f in FEATURES], float)


def _name(centroids: np.ndarray) -> list[tuple[str, str]]:
    """A name and a one-line meaning per segment: each name scores segments by one feature (z-score
    across segment centres, signed), and names are matched one-to-one to maximise the total score."""
    from scipy.optimize import linear_sum_assignment

    mu, sd = centroids.mean(0), centroids.std(0) + 1e-9
    z = (centroids - mu) / sd
    ix = {f: i for i, f in enumerate(FEATURES)}
    names = [("Steady seller", "volatility", -1, "regular weeks with small swings and almost no empty weeks"),
             ("Declining seller", "trend", -1, "sales have been falling over the last half year"),
             ("Growing seller", "trend", 1, "sales have been climbing over the last half year"),
             ("Intermittent seller", "intermittency", 1, "sells in bursts, with many weeks of no sales"),
             ("Seasonal peaker", "season_concentration", 1, "a large part of the year's sales lands in a few weeks"),
             ("Moderate seller", None, 0, "between the other groups: noticeable swings, few empty weeks")]
    score = np.array([[0.3 if f is None else sign * row[ix[f]] for _, f, sign, _ in names] for row in z])
    rows, cols = linear_sum_assignment(-score)
    out = [None] * len(z)
    for r, c in zip(rows, cols):
        out[r] = (names[c][0], names[c][3])
    return out


def fit(series: dict[str, np.ndarray], origin: dict[str, int]) -> dict:
    """Fit the segmentation on the panel's feature window ending at `origin` (no later data is used)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    ids = [s for s, y in series.items() if origin[s] >= WINDOW]
    raw = np.array([_vector(profile(series[s][:origin[s]])) for s in ids])
    lo, hi = np.quantile(raw, WINSOR[0], axis=0), np.quantile(raw, WINSOR[1], axis=0)
    scaler = StandardScaler().fit(np.clip(raw, lo, hi))
    X = scaler.transform(np.clip(raw, lo, hi))
    rows = []
    for k in K_RANGE:
        labs = [KMeans(k, n_init=10, random_state=s).fit_predict(X) for s in range(5)]
        stab = float(np.mean([adjusted_rand_score(a, b) for a, b in combinations(labs, 2)]))
        rows.append({"k": k, "silhouette": float(silhouette_score(X, labs[0])), "stability_ari": stab})
    ok = [r for r in rows if r["stability_ari"] >= 0.8] or rows
    k = max(ok, key=lambda r: r["silhouette"])["k"]
    km = KMeans(k, n_init=10, random_state=0).fit(X)
    names = _name(km.cluster_centers_)
    dist = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
    segs = []
    for c in range(k):
        m = km.labels_ == c
        segs.append({"id": c, "name": names[c][0], "meaning": names[c][1], "members": int(m.sum()),
                     "share_ross": float(np.mean([ids[i].startswith("ross_") for i in np.where(m)[0]])),
                     "centre": dict(zip(FEATURES, raw[m].mean(0).round(4))), "benchmark": {}})
    return {"features": FEATURES, "lo": lo, "hi": hi, "scaler": scaler, "km": km, "segments": segs,
            "distance_p95": float(np.quantile(dist, 0.95)), "distance_scale": float(np.median(dist)),
            "selection": rows, "ids": ids, "labels": km.labels_, "k": k}


def benchmark(model: dict, series: dict[str, np.ndarray], weeks, origin: dict[str, int], tcn) -> None:
    """Error of Auto and of the TCN on each segment's members at the validation origin (MASE; lower is better)."""
    from startupiq.dl import windows as W
    from startupiq.dl.train import predict
    from startupiq.evaluate.metrics import mase, mase_scale
    from startupiq.features import weekly as fw
    from startupiq.features.splits import WEEK_HORIZON as H
    from startupiq.models.forecast_methods import auto

    ids = model["ids"]
    ev = W.build({s: series[s] for s in ids}, {s: weeks[s] for s in ids}, {s: [origin[s]] for s in ids})
    q = {sid: p for sid, p in zip(ev.series, predict(tcn, ev))}
    per = {}
    for sid in ids:
        y, o = series[sid], origin[sid]
        scale = mase_scale(y[:o], fw.SEASON)
        a = mase(y[o:o + H], auto(y[:o], H, fw.SEASON)[0], scale)
        t = mase(y[o:o + H], q[sid][:, 1], scale)
        if np.isfinite(a) and np.isfinite(t):
            per[sid] = (a, t)
    for seg in model["segments"]:
        mem = [per[s] for s, c in zip(ids, model["labels"]) if c == seg["id"] and s in per]
        a, t = np.array([m[0] for m in mem]), np.array([m[1] for m in mem])
        seg["benchmark"] = {"series": len(mem), "auto_mase": round(float(a.mean()), 3), "tcn_mase": round(float(t.mean()), 3),
                            "tcn_wins": round(float(np.mean(t < a)), 3)}


@lru_cache(maxsize=1)
def load() -> dict | None:
    return joblib.load(MODEL) if MODEL.exists() else None


def assign(y) -> dict | None:
    """Segment of a weekly series: name, confidence, how typical it is, and its features against the segment's centre."""
    m = load()
    if m is None or len(y) < WINDOW:
        return None
    p = profile(y)
    v = np.clip(_vector(p), m["lo"], m["hi"])
    x = m["scaler"].transform(v[None])[0]
    d = np.linalg.norm(m["km"].cluster_centers_ - x, axis=1)
    w = np.exp(-(d - d.min()) / (m["distance_scale"] * 0.5))
    share = w / w.sum()
    best = int(d.argmin())
    seg = m["segments"][best]
    runner = int(np.argsort(d)[1])
    return {"segment": seg["name"], "meaning": seg["meaning"], "confidence": round(float(share[best]), 2),
            "runner_up": m["segments"][runner]["name"], "runner_up_share": round(float(share[runner]), 2),
            "unusual": bool(d[best] > m["distance_p95"]), "peers": seg["members"], "benchmark": seg["benchmark"],
            "features": [{"key": f, "label": LABELS[f], "yours": round(p[f], 3), "segment": seg["centre"][f]} for f in FEATURES],
            "all": [{"segment": s["name"], "share": round(float(share[i]), 2)} for i, s in enumerate(m["segments"])]}


def main() -> None:
    import json

    import torch

    from startupiq.dl.train import Config, build_model
    from startupiq.dl import windows as W
    from startupiq.evaluate import report
    from startupiq.features import weekly as fw
    from startupiq.features.splits import WEEK_HORIZON as H

    series, weeks = fw.panel()
    origin = {s: len(y) - 2 * H for s, y in series.items()}  # the validation origin; the test windows are never touched
    model = fit(series, origin)
    cfg = Config(**json.loads(Path("docs/results/dl_weekly.json").read_text(encoding="utf-8"))["chosen"]["tcn"]["config"])
    sid = next(iter(series))
    dummy = W.build({sid: series[sid]}, {sid: weeks[sid]}, {sid: [origin[sid]]})
    tcn = build_model(cfg, dummy, W.L)
    tcn.load_state_dict(torch.load("data/models/dl_weekly_tcn.pt", weights_only=True, map_location=next(tcn.parameters()).device))  # trained on targets before this origin
    tcn.eval()
    benchmark(model, series, weeks, origin, tcn)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL)
    segs = [{"segment": s["name"], "series": s["members"], **s["benchmark"], **{k: v for k, v in s["centre"].items()}} for s in model["segments"]]
    import pandas as pd
    report.write("segments", "Business segments (the profiling stage)", [
        ("Data", f"{len(model['ids'])} public weekly series (Rossmann stores, UCI products); features from the 52 weeks before the validation origin."),
        ("Choice of k", report.table(pd.DataFrame(model["selection"]).round(3)) + f"\n\nChosen k = {model['k']}: highest silhouette with stability ARI >= 0.8."),
        ("Segments, with MASE of Auto and the TCN on each segment's members (validation origin; lower is better)", report.table(pd.DataFrame(segs))),
    ], {"selection": model["selection"], "segments": segs}, command="python -m startupiq.segment")
    print(pd.DataFrame(segs).to_string())


if __name__ == "__main__":
    main()
