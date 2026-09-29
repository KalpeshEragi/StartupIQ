"""Task 5 entry point: cleaned data -> engineered features -> split -> selection -> ML-ready matrices.

Run from the StartupIQ folder:  python Task5/run_task5.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.build import FEATURE_SETS, build_features, registry_frame
from src.features.select import make_preprocessor, select_features

SEED = 42
CLEAN = ROOT / "data" / "processed" / "startups_clean.csv"
OUT = ROOT / "Task5" / "outputs"
ART = ROOT / "models"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(exist_ok=True)
    df = pd.read_csv(CLEAN, parse_dates=["founded_at", "first_funding_at", "last_funding_at"])
    feats = build_features(df)
    registry_frame().to_csv(OUT / "feature_registry.csv", index=False)

    # modelling population = resolved outcomes (acquired / closed); operating is unresolved
    frame = feats[feats.target.notna()].copy()
    frame["target"] = frame.target.astype(int)
    train_idx, test_idx = train_test_split(frame.index, test_size=0.2, stratify=frame.target, random_state=SEED)
    frame["split"] = "train"
    frame.loc[test_idx, "split"] = "test"
    frame.to_csv(ROOT / "data" / "processed" / "model_frame.csv", index=False)
    tr, te = frame[frame.split == "train"], frame[frame.split == "test"]
    print(f"resolved={len(frame)} train={len(tr)} test={len(te)} acquired share train={tr.target.mean():.3f} test={te.target.mean():.3f}")

    selected, summary = {}, {}
    for name, cand in FEATURE_SETS.items():
        keep, rep, imp = select_features(tr, tr.target, cand, seed=SEED)
        selected[name] = keep
        rep.assign(feature_set=name).to_csv(OUT / f"selection_dropped_{name}.csv", index=False)
        imp.to_csv(OUT / f"importance_{name}.csv", index=False)
        summary[name] = {"candidates": len(cand), "selected": len(keep), "features": keep}
        print(f"{name}: {len(cand)} -> {len(keep)}")
        if len(rep):
            print(rep[["feature", "reason", "detail"]].to_string(index=False))

    (OUT / "selected_features.json").write_text(json.dumps(selected, indent=2))

    # ML-ready matrices (preprocessing fitted on train only) for the two headline sets
    for name in ("early_safe", "full_history"):
        pre = make_preprocessor(selected[name]).fit(tr[selected[name]])
        joblib.dump(pre, ART / f"preprocessor_{name}.joblib")
        for label, part in (("train", tr), ("test", te)):
            X = pd.DataFrame(pre.transform(part[selected[name]]), columns=pre.get_feature_names_out(), index=part.index)
            X.assign(target=part.target).to_csv(ROOT / "data" / "processed" / f"X_{name}_{label}.csv", index=False)
        print(name, "matrix width:", len(pre.get_feature_names_out()))


if __name__ == "__main__":
    main()
