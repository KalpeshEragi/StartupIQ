"""Task 5 feature selection + preprocessing factory.

All learned steps (imputation, scaling, encoding vocabulary, selection statistics) are
fitted on training data only.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build import FEATURE_REGISTRY


def split_types(features):
    cat = [f for f in features if FEATURE_REGISTRY[f][1] == "cat"]
    num = [f for f in features if FEATURE_REGISTRY[f][1] == "num"]
    return num, cat


def make_preprocessor(features, min_frequency=50) -> ColumnTransformer:
    num, cat = split_types(features)
    parts = []
    if num:
        parts.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler())]), num))
    if cat:
        parts.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                                     sparse_output=False))]), cat))
    return ColumnTransformer(parts, verbose_feature_names_out=False)


def select_features(X: pd.DataFrame, y: pd.Series, candidates, missing_max=0.6, dominant_max=0.995,
                    corr_max=0.9, perm_min=0.001, seed=42):
    """Returns (selected list, report DataFrame). Uses training data only."""
    report = []
    keep = list(candidates)

    def drop(f, why, detail=""):
        keep.remove(f)
        report.append({"feature": f, "action": "dropped", "reason": why, "detail": detail})

    # 1. missingness
    for f in list(keep):
        m = X[f].isna().mean()
        if m > missing_max:
            drop(f, "too many missing", f"{m:.0%}")
    # 2. near-constant
    for f in list(keep):
        top = X[f].value_counts(normalize=True, dropna=True)
        if len(top) <= 1 or top.iloc[0] > dominant_max:
            drop(f, "near-constant", f"top value share {top.iloc[0] if len(top) else 1:.3f}")
    # 3. redundancy between numeric features (keep the one more associated with the target)
    num, _ = split_types(keep)
    if len(num) > 1:
        corr = X[num].corr(method="spearman").abs()
        assoc = X[num].apply(lambda c: abs(c.corr(y, method="spearman")))
        for i, a in enumerate(num):
            for b in num[i + 1:]:
                if a in keep and b in keep and corr.loc[a, b] > corr_max:
                    loser = a if assoc[a] < assoc[b] else b
                    drop(loser, "redundant", f"|spearman|={corr.loc[a, b]:.2f} with {b if loser == a else a}")
    # 4. model-based: permutation importance on a validation slice of the TRAIN data
    Xa, Xv, ya, yv = train_test_split(X[keep], y, test_size=0.25, stratify=y, random_state=seed)
    pipe = Pipeline([("prep", make_preprocessor(keep)),
                     ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=10, class_weight="balanced",
                                                   n_jobs=-1, random_state=seed))]).fit(Xa, ya)
    pi = permutation_importance(pipe, Xv, yv, scoring="roc_auc", n_repeats=10, random_state=seed, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=keep)
    for f in list(keep):
        if imp[f] < perm_min:
            drop(f, "low permutation importance", f"{imp[f]:.4f} AUC")
    imp_df = pd.DataFrame({"feature": imp.index, "perm_importance_auc": imp.values,
                           "perm_std": pi.importances_std}).sort_values("perm_importance_auc", ascending=False)
    rep = pd.DataFrame(report)
    return keep, rep, imp_df
