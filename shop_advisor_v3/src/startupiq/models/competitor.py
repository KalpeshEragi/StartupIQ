"""Task 6 competitor-move classifier on Olist (decision 0006).

Predict a seller's next-month move (exit, price cut, expansion, stagnate) from what is
known at month t. Primary metric macro-F1; Brier score for probability quality.
Validation picks the model; --final refits on train+val and scores test once.
"""

from __future__ import annotations

import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_text

from startupiq.evaluate import report
from startupiq.evaluate.metrics import brier_multiclass, macro_f1, mean_sd
from startupiq.features import olist as fo
from startupiq.features.splits import SEEDS

CLASSES = sorted(fo.LABELS)


def _pre(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), categorical),
    ])


def _tree_pre(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), categorical),
    ])


def make_models(seed: int, numeric=fo.NUMERIC, categorical=fo.CATEGORICAL) -> dict:
    logit = make_pipeline(_pre(numeric, categorical), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
    svm = make_pipeline(_pre(numeric, categorical), CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=0.1, random_state=seed), cv=3))
    tree = make_pipeline(_tree_pre(numeric, categorical), DecisionTreeClassifier(max_depth=6, min_samples_leaf=40, class_weight="balanced", random_state=seed))
    rf = make_pipeline(_tree_pre(numeric, categorical), RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=seed))
    gbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, class_weight="balanced", random_state=seed, verbose=-1)
    vote = VotingClassifier([("lr", logit), ("rf", rf)], voting="soft")
    return {"Logistic regression": logit, "Linear SVM (calibrated)": svm, "Decision tree (depth 6)": tree,
            "Random forest": rf, "LightGBM": gbm, "Soft vote (logistic + forest)": vote}


def _fit_predict(model, Xtr, ytr, Xev):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xev)
    order = [list(model.classes_).index(c) for c in CLASSES]
    proba = proba[:, order]
    return np.array(CLASSES)[proba.argmax(1)], proba


def _evaluate(rows: pd.DataFrame, train_splits: list[str], eval_split: str, seeds=SEEDS, features=fo.FEATURES):
    tr, ev = rows[rows.split.isin(train_splits)], rows[rows.split == eval_split]
    numeric = [f for f in features if f in fo.NUMERIC]
    categorical = [f for f in features if f in fo.CATEGORICAL]
    results, keep = [], {}
    majority = tr.label.value_counts().idxmax()
    results.append({"model": f"Majority class ({majority})", "macro_f1": [macro_f1(ev.label, [majority] * len(ev))],
                    "accuracy": [accuracy_score(ev.label, [majority] * len(ev))], "brier": [np.nan]})
    results.append({"model": "Same move as last month", "macro_f1": [macro_f1(ev.label, ev.label_prev)],
                    "accuracy": [accuracy_score(ev.label, ev.label_prev)], "brier": [np.nan]})
    for name in make_models(0, numeric, categorical):
        f1s, accs, briers = [], [], []
        deterministic = name.startswith("Logistic")
        for seed in (seeds[:1] if deterministic else seeds):
            model = make_models(seed, numeric, categorical)[name]
            pred, proba = _fit_predict(model, tr[features], tr.label, ev[features])
            f1s.append(macro_f1(ev.label, pred)); accs.append(accuracy_score(ev.label, pred))
            briers.append(brier_multiclass(ev.label.to_numpy(), proba, CLASSES))
            keep.setdefault(name, (model, pred, proba))
        results.append({"model": name, "macro_f1": f1s, "accuracy": accs, "brier": briers})
    return results, keep, tr, ev


def _table(results) -> pd.DataFrame:
    return pd.DataFrame([{"model": r["model"], "macro-F1": mean_sd(r["macro_f1"]), "accuracy": mean_sd(r["accuracy"]),
                          "Brier": "n/a" if np.isnan(r["brier"][0]) else mean_sd(r["brier"])} for r in results])


def _choose(results) -> str:
    """Decision rule (0006): best macro-F1 within one sd; ties -> lower Brier; then simpler (earlier in list)."""
    learned = [r for r in results if not np.isnan(r["brier"][0])]
    best = max(learned, key=lambda r: np.mean(r["macro_f1"]))
    tol = max(np.std(best["macro_f1"]), 1e-9)
    tied = [r for r in learned if np.mean(best["macro_f1"]) - np.mean(r["macro_f1"]) <= tol]
    return min(tied, key=lambda r: np.mean(r["brier"]))["model"]


def run(final: bool = False) -> dict:
    t0 = time.time()
    rows = fo.build()
    out: dict = {"rows": rows.split.value_counts().to_dict(),
                 "class_share": rows.groupby("split").label.value_counts(normalize=True).round(3).unstack().to_dict()}

    results, kept, tr, ev = _evaluate(rows, ["train"], "val")
    chosen = _choose(results)
    out["validation"] = _table(results).to_dict("records")
    out["chosen"] = chosen
    model, pred, proba = kept[chosen]
    cm = pd.DataFrame(confusion_matrix(ev.label, pred, labels=CLASSES), index=[f"true {c}" for c in CLASSES], columns=[f"pred {c}" for c in CLASSES])
    per_class = classification_report(ev.label, pred, labels=CLASSES, zero_division=0)

    # Feature selection (Task 5): mutual information ranking, then the chosen model on the top half.
    num = SimpleImputer(strategy="median").fit_transform(tr[fo.NUMERIC])
    mi = pd.Series(mutual_info_classif(num, tr.label, random_state=0), index=fo.NUMERIC).sort_values(ascending=False)
    top = list(mi.index[: len(fo.NUMERIC) // 2]) + fo.CATEGORICAL
    sel_results, _, _, _ = _evaluate(rows, ["train"], "val", seeds=SEEDS[:1], features=top)
    sel_f1 = next(np.mean(r["macro_f1"]) for r in sel_results if r["model"] == chosen)
    lgb_model = kept["LightGBM"][0]
    perm = permutation_importance(lgb_model, ev[fo.FEATURES], ev.label, scoring="f1_macro", n_repeats=5, random_state=0)
    perm_s = pd.Series(perm.importances_mean, index=fo.FEATURES).sort_values(ascending=False)
    importance = pd.DataFrame({"feature": mi.index, "mutual_information": mi.values.round(4)}).merge(
        pd.DataFrame({"feature": perm_s.index, "lightgbm_permutation_macro_f1_drop": perm_s.values.round(4)}), on="feature", how="outer")
    importance = importance.sort_values("lightgbm_permutation_macro_f1_drop", ascending=False)

    tree = make_models(0)["Decision tree (depth 6)"]
    tree.fit(tr[fo.FEATURES], tr.label)
    names = list(tree[0].get_feature_names_out())
    rules = export_text(tree[-1], feature_names=names, max_depth=3)

    # Error analysis: where does the chosen model go wrong?
    ev = ev.assign(pred=pred, correct=pred == ev.label)
    ev["size_band"] = pd.qcut(ev.log_revenue, 3, labels=["small", "medium", "large"])
    errors = {
        "accuracy_by_size_band": ev.groupby("size_band", observed=True).correct.mean().round(3).to_dict(),
        "macro_f1_by_size_band": {b: round(macro_f1(g.label, g.pred), 3) for b, g in ev.groupby("size_band", observed=True)},
        "accuracy_by_active_months": ev.groupby(pd.cut(ev.active_months, [0, 3, 6, 12, 30]), observed=True).correct.mean().round(3).astype(float).to_dict(),
        "top_confusions": (ev[~ev.correct].groupby(["label", "pred"]).size().sort_values(ascending=False).head(5).to_dict()),
    }
    out["errors"] = {k: {str(kk): vv for kk, vv in v.items()} for k, v in errors.items()}

    sections = [
        ("Data", f"Rows (seller-category-months): {out['rows']}. Class shares per split:\n\n" +
         report.table(rows.groupby("split").label.value_counts(normalize=True).round(3).unstack().reset_index(), "{:.3f}")),
        ("Validation results (3 seeds where the model is random; mean ± sd)", report.table(_table(results))),
        ("Chosen by the decision rule", f"**{chosen}** (best macro-F1 within one sd, then lowest Brier)."),
        ("Confusion matrix (validation, chosen model)", report.table(cm.reset_index().rename(columns={"index": ""}))),
        ("Per-class precision and recall (validation)", "```\n" + per_class + "\n```"),
        ("Feature importance", report.table(importance)),
        ("Feature selection", f"Top {len(top) - len(fo.CATEGORICAL)} numeric features by mutual information (+ categories): "
                              f"{chosen} macro-F1 {sel_f1:.4f} vs {np.mean(next(r['macro_f1'] for r in results if r['model'] == chosen)):.4f} with all features."),
        ("Readable rules (decision tree, top 3 levels)", "```\n" + rules + "\n```"),
        ("Error analysis (validation)", "```\n" + "\n".join(f"{k}: {v}" for k, v in out["errors"].items()) + "\n```"),
    ]
    if final:
        test_results, test_kept, _, te = _evaluate(rows, ["train", "val"], "test")
        out["test"] = _table(test_results).to_dict("records")
        _, tpred, _ = test_kept[chosen]
        tcm = pd.DataFrame(confusion_matrix(te.label, tpred, labels=CLASSES), index=[f"true {c}" for c in CLASSES], columns=[f"pred {c}" for c in CLASSES])
        sections.insert(3, ("Test results (refit on train + validation, scored once)", report.table(_table(test_results))))
        sections.insert(4, ("Confusion matrix (test, chosen model)", report.table(tcm.reset_index().rename(columns={"index": ""}))))
    out["seconds"] = round(time.time() - t0)
    report.write("competitor", "Competitor-move classifier (Tasks 5 and 6)", sections, out)
    return out
