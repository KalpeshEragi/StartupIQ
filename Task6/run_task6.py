"""Task 6 entry point: train, validate, test and analyse outcome models (acquired vs closed).

Run from the StartupIQ folder:  python Task6/run_task6.py   (needs Task 5 output)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from src.evaluation.metrics import error_rate_by, score
from src.features.build import FEATURE_SETS
from src.models.train import MODELS, SEED, make_pipeline

OUT = ROOT / "Task6" / "outputs"
ART = ROOT / "models"
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)
SETS = ["profile_only", "early_safe", "early_safe_plus_calendar", "full_history"]
GB_SETS = {"early_safe", "full_history"}  # boosted trees only where the comparison matters


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    ART.mkdir(exist_ok=True)
    frame = pd.read_csv(ROOT / "data" / "processed" / "model_frame.csv")
    selected = json.loads((ROOT / "Task5" / "outputs" / "selected_features.json").read_text())
    tr, te = frame[frame.split == "train"], frame[frame.split == "test"]
    ytr, yte = tr.target, te.target

    rows, fitted, test_proba = [], {}, {}

    # trivial reference
    d = DummyClassifier(strategy="prior").fit(tr[["age_at_first_funding"]], ytr)
    rows.append({"feature_set": "-", "model": "majority_class", **{f"test_{k}": v for k, v in
                 score(yte, d.predict_proba(te[["age_at_first_funding"]])[:, 1]).items()}})

    for fs in SETS:
        feats = selected[fs]
        for name in MODELS:
            if name == "hist_gradient_boosting" and fs not in GB_SETS:
                continue
            pipe, grid, purpose = make_pipeline(name, feats)
            gs = GridSearchCV(pipe, grid, cv=CV, scoring="roc_auc", refit=True, n_jobs=1).fit(tr[feats], ytr)
            best = gs.best_estimator_
            cv = cross_validate(clone(best), tr[feats], ytr, cv=CV, scoring=["roc_auc", "f1_macro"],
                                return_train_score=True, n_jobs=1)
            proba_te = best.predict_proba(te[feats])[:, 1]
            proba_tr = best.predict_proba(tr[feats])[:, 1]
            row = {
                "feature_set": fs, "model": name, "purpose": purpose, "n_features": len(feats),
                "best_params": json.dumps({k.replace("model__", ""): v for k, v in gs.best_params_.items()}),
                "train_roc_auc": score(ytr, proba_tr)["roc_auc"], "train_macro_f1": score(ytr, proba_tr)["macro_f1"],
                "cv_roc_auc": cv["test_roc_auc"].mean(), "cv_roc_auc_std": cv["test_roc_auc"].std(),
                "cv_macro_f1": cv["test_f1_macro"].mean(),
                **{f"test_{k}": v for k, v in score(yte, proba_te).items()},
            }
            rows.append(row)
            fitted[(fs, name)], test_proba[(fs, name)] = best, proba_te
            print(f"{fs:26s} {name:24s} train_auc={row['train_roc_auc']:.3f} cv_auc={row['cv_roc_auc']:.3f} "
                  f"test_auc={row['test_roc_auc']:.3f} test_macroF1={row['test_macro_f1']:.3f}")

    res = pd.DataFrame(rows)
    res["auc_gap_train_minus_cv"] = res.train_roc_auc - res.cv_roc_auc
    res["auc_gap_cv_minus_test"] = res.cv_roc_auc - res.test_roc_auc
    res.round(4).to_csv(OUT / "model_results.csv", index=False)

    # model choice per set = best CV ROC-AUC (never the test score)
    ranked = res[res.model != "majority_class"]
    chosen = {fs: ranked[ranked.feature_set == fs].sort_values("cv_roc_auc").iloc[-1].model for fs in SETS}
    (OUT / "chosen_models.json").write_text(json.dumps(chosen, indent=2))
    print("chosen:", chosen)

    early_name, full_name = chosen["early_safe"], chosen["full_history"]
    early, full = fitted[("early_safe", early_name)], fitted[("full_history", full_name)]
    joblib.dump(early, ART / "outcome_model_early_safe.joblib")
    joblib.dump(full, ART / "outcome_model_full_history.joblib")
    (ART / "outcome_model_early_safe_features.json").write_text(json.dumps(selected["early_safe"]))

    # ---- figures ---------------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (label, key) in zip(axs, [("early_safe", ("early_safe", early_name)), ("full_history", ("full_history", full_name))]):
        pred = (test_proba[key] >= .5).astype(int)
        ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=["closed", "acquired"], ax=ax, colorbar=False)
        ax.set_title(f"{label}: {key[1]}")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "confusion_matrices.png", dpi=110); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for label, key in [("early_safe", ("early_safe", early_name)), ("full_history", ("full_history", full_name)),
                       ("profile_only", ("profile_only", chosen["profile_only"]))]:
        RocCurveDisplay.from_predictions(yte, test_proba[key], name=label, ax=axs[0])
        PrecisionRecallDisplay.from_predictions(yte, test_proba[key], name=label, ax=axs[1])
    axs[0].set_title("ROC (test)"); axs[1].set_title("Precision-recall (test)")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "roc_pr_curves.png", dpi=110); plt.close(fig)

    cmp = ranked[ranked.apply(lambda r: chosen[r.feature_set] == r.model, axis=1)].set_index("feature_set").loc[SETS]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(SETS)); w = .27
    ax.bar(x - w, cmp.train_roc_auc, w, label="train AUC"); ax.bar(x, cmp.cv_roc_auc, w, label="CV AUC")
    ax.bar(x + w, cmp.test_roc_auc, w, label="test AUC")
    ax.set_xticks(x, [s.replace("_", "\n") for s in SETS], fontsize=8); ax.set_ylim(.5, 1.02); ax.legend()
    ax.set_title("Train vs CV vs test ROC-AUC (chosen model per feature set)")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "feature_set_comparison.png", dpi=110); plt.close(fig)

    # ---- permutation importance of the primary (early_safe) model on the test set -----
    pi = permutation_importance(early, te[selected["early_safe"]], yte, scoring="roc_auc", n_repeats=15, random_state=SEED)
    pd.DataFrame({"feature": selected["early_safe"], "auc_drop": pi.importances_mean, "std": pi.importances_std}
                 ).sort_values("auc_drop", ascending=False).round(4).to_csv(OUT / "early_safe_test_importance.csv", index=False)

    # ---- error analysis on the primary model -----------------------------------------
    ea = te.copy()
    ea["proba"] = test_proba[("early_safe", early_name)]
    ea["pred"] = (ea.proba >= .5).astype(int)
    ea["full_pred"] = (test_proba[("full_history", full_name)] >= .5).astype(int)
    ea["age_missing"] = ea.age_at_first_funding.isna()
    ea["age_bin"] = pd.cut(ea.age_at_first_funding, [-.1, 0.5, 2, 5, 100], labels=["<=0.5y", "0.5-2y", "2-5y", ">5y"])
    ea["seed_flag"] = ea.has_seed.map({0.0: "no seed", 1.0: "seed"}).fillna("unknown funding")
    ea["founded_missing"] = ea.founded_year.isna()
    ea["founded_era"] = pd.cut(ea.founded_year, [1900, 2000, 2005, 2009, 2015], labels=["<=2000", "2001-05", "2006-09", "2010+"])
    ea["outcome_error"] = np.select([(ea.pred == 1) & (ea.target == 0), (ea.pred == 0) & (ea.target == 1)],
                                    ["false_positive (pred acquired, was closed)", "false_negative (pred closed, was acquired)"], "correct")
    tables = {}
    for col in ["market", "country_code", "age_bin", "age_missing", "seed_flag", "founded_era", "founded_missing"]:
        t = error_rate_by(ea, col)
        t.to_csv(OUT / f"errors_by_{col}.csv")
        tables[col] = t
    ea.outcome_error.value_counts().to_csv(OUT / "error_counts.csv")
    cols = ["name", "market", "country_code", "status", "proba", "age_at_first_funding", "has_seed", "has_angel"]
    ea[ea.outcome_error.str.startswith("false_pos")].nlargest(12, "proba")[cols].round(3).to_csv(OUT / "top_false_positives.csv", index=False)
    ea[ea.outcome_error.str.startswith("false_neg")].nsmallest(12, "proba")[cols].round(3).to_csv(OUT / "top_false_negatives.csv", index=False)
    both = pd.crosstab(ea.pred == ea.target, ea.full_pred == ea.target, rownames=["early_correct"], colnames=["full_correct"])
    both.to_csv(OUT / "early_vs_full_agreement.csv")
    print("\nerror counts:\n", ea.outcome_error.value_counts())
    print("\nearly vs full correctness:\n", both)
    for k in ("age_bin", "seed_flag", "founded_era", "age_missing"):
        print(f"\nerrors by {k}\n", tables[k])

    # ---- time-based robustness check (early_safe features, no calendar features) ---------
    feats = selected["early_safe"]
    dated = frame[frame.founded_year_meta.notna()]
    a, b = dated[dated.founded_year_meta <= 2007], dated[dated.founded_year_meta >= 2008]
    m = clone(early).fit(a[feats], a.target)
    temporal = {"train_founded<=2007": len(a), "test_founded>=2008": len(b),
                "acquired_share_train": round(a.target.mean(), 3), "acquired_share_test": round(b.target.mean(), 3),
                **{k: round(float(v), 4) for k, v in score(b.target, m.predict_proba(b[feats])[:, 1]).items() if k in ("roc_auc", "pr_auc", "macro_f1", "recall", "precision")}}
    (OUT / "temporal_check.json").write_text(json.dumps(temporal, indent=2))
    print("\ntemporal check:", temporal)


if __name__ == "__main__":
    main()
