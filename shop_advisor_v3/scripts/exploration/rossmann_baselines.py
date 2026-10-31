"""Exploration script (2026-10-23/24). Run from the project root. Numbers are logged in devdocs/evaluation_log.md."""
import pandas as pd, numpy as np
root = "data/raw"
tr = pd.read_csv(root + r"\rossmann_hf\train.csv", low_memory=False, parse_dates=["Date"])
piv = tr.pivot(index="Date", columns="Store", values="Sales")
last = tr.Date.max(); cut = last - pd.Timedelta(days=41)
idx = piv.index[piv.index >= cut]
act = piv.loc[idx].astype(float)
hist = piv[piv.index < cut].astype(float).replace(0, np.nan)
def rmspe(a, p):
    a = a.values.ravel(); p = np.asarray(p, dtype=float).ravel()
    m = (a > 0) & np.isfinite(p)
    return float(np.sqrt(np.mean(((a[m] - p[m]) / a[m]) ** 2))), int(m.sum())
mean_pred = np.tile(hist.mean().values, (len(idx), 1))
wd = np.vstack([hist[hist.index.dayofweek == d.dayofweek].tail(4).mean().values for d in idx])
ly = piv.shift(364).loc[idx].astype(float).replace(0, np.nan).values
print("train ends", (cut - pd.Timedelta(days=1)).date(), "test", idx[0].date(), "to", idx[-1].date())
for n, p in [("store mean", mean_pred), ("same weekday, last 4 weeks", wd), ("same day last year", ly)]:
    print(n, rmspe(act, p))
