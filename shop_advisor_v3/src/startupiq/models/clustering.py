"""Task 6 unsupervised: segment Rossmann stores by sales behaviour ("businesses like yours").

Features use the training window only (before the test start) and describe behaviour a
founder's upload can also give: level, volatility, promotion response, December peak,
weekend share, trend. K-means and Ward hierarchical clustering compared on silhouette,
Davies-Bouldin, stability (ARI across seeds / bootstrap) and agreement with StoreType.
"""

from __future__ import annotations

import time
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from startupiq.evaluate import report
from startupiq.features.splits import ROSS_TEST_START

K_RANGE = range(2, 9)
FEATURES = ["log_weekly_sales", "weekly_cv", "promo_lift", "december_ratio", "saturday_share", "trend_ratio"]
WINSOR = (0.01, 0.99)  # cap extreme stores (e.g. ones that opened inside the window) so they do not form their own cluster


def store_profiles() -> pd.DataFrame:
    d = pd.read_parquet("data/processed/rossmann_daily.parquet")
    d = d[(d.Date < ROSS_TEST_START) & (d.Open == 1) & (d.Sales > 0)]
    w = pd.read_parquet("data/processed/rossmann_weekly.parquet")
    w = w[w.week < ROSS_TEST_START]
    g = w.groupby("Store").sales
    prof = pd.DataFrame({"log_weekly_sales": np.log(g.mean()), "weekly_cv": g.std() / g.mean()})
    promo = d.groupby(["Store", "Promo"]).Sales.mean().unstack()
    prof["promo_lift"] = promo[1] / promo[0]
    dec = d.assign(dec=d.Date.dt.month == 12).groupby(["Store", "dec"]).Sales.mean().unstack()
    prof["december_ratio"] = dec[True] / dec[False]
    week_tot = d.groupby("Store").Sales.sum()
    prof["saturday_share"] = d[d.DayOfWeek == 6].groupby("Store").Sales.sum() / week_tot
    first = w[w.week < w.week.min() + pd.Timedelta(weeks=26)].groupby("Store").sales.mean()
    last = w[w.week >= ROSS_TEST_START - pd.Timedelta(weeks=26)].groupby("Store").sales.mean()
    prof["trend_ratio"] = np.log(last / first)  # log: a store that doubled and one that halved are equally far from 1
    store = d.drop_duplicates("Store").set_index("Store")[["StoreType", "Assortment"]]
    return prof.join(store).dropna()


def run(final: bool = False) -> dict:  # no test split: unsupervised, scored on internal criteria
    t0 = time.time()
    prof = store_profiles()
    capped = prof[FEATURES].clip(prof[FEATURES].quantile(WINSOR[0]), prof[FEATURES].quantile(WINSOR[1]), axis=1)
    X = StandardScaler().fit_transform(capped)
    rows = []
    for k in K_RANGE:
        km = KMeans(k, n_init=10, random_state=0).fit_predict(X)
        ward = AgglomerativeClustering(k, linkage="ward").fit_predict(X)
        seeds = [KMeans(k, n_init=10, random_state=s).fit_predict(X) for s in range(5)]
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(5):
            idx = rng.choice(len(X), int(0.8 * len(X)), replace=False)
            boot.append(adjusted_rand_score(ward[idx], AgglomerativeClustering(k, linkage="ward").fit_predict(X[idx])))
        for name, lab, stab in (("K-means", km, np.mean([adjusted_rand_score(a, b) for a, b in combinations(seeds, 2)])),
                                ("Ward", ward, np.mean(boot))):
            rows.append({"method": name, "k": k, "silhouette": silhouette_score(X, lab), "davies_bouldin": davies_bouldin_score(X, lab),
                         "stability_ARI": stab, "ARI_vs_StoreType": adjusted_rand_score(prof.StoreType, lab),
                         "NMI_vs_StoreType": normalized_mutual_info_score(prof.StoreType, lab)})
    res = pd.DataFrame(rows)
    # Choose: highest silhouette among k >= 3 (k = 2 rarely gives useful peer groups) with stability >= 0.8.
    ok = res[(res.k >= 3) & (res.stability_ARI >= 0.8)]
    best = (ok if len(ok) else res).sort_values("silhouette", ascending=False).iloc[0]
    k, method = int(best.k), best.method
    labels = (KMeans(k, n_init=10, random_state=0).fit_predict(X) if method == "K-means"
              else AgglomerativeClustering(k, linkage="ward").fit_predict(X))
    prof["cluster"] = labels
    profile = prof.groupby("cluster")[FEATURES].mean().round(3)
    profile.insert(0, "stores", prof.groupby("cluster").size())
    profile["weekly_sales"] = np.exp(profile.log_weekly_sales).round(0)
    mix = pd.crosstab(prof.cluster, prof.StoreType)
    prof[["cluster"]].to_parquet("data/processed/rossmann_store_clusters.parquet")
    out = {"table": res.round(4).to_dict("records"), "chosen": {"method": method, "k": k}, "profile": profile.reset_index().to_dict("records"),
           "seconds": round(time.time() - t0)}
    sections = [
        ("Data", f"{len(prof)} stores; features from the training window only: {', '.join(FEATURES)} (trend as log ratio; capped at the 1st and 99th percentile; standardised)."),
        ("K-means vs Ward across k", report.table(res.round(4))),
        ("Chosen", f"**{method}, k = {k}**: highest silhouette among k >= 3 with stability ARI >= 0.8."),
        ("Cluster profiles (means)", report.table(profile.reset_index())),
        ("Clusters vs StoreType (counts)", report.table(mix.reset_index())),
    ]
    report.write("clustering", "Store segments (Task 6, unsupervised)", sections, out)
    return out
