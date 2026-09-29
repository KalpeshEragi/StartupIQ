"""Task 4 entry point: raw -> cleaned dataset + EDA outputs.

Run from the StartupIQ folder:  python Task4/run_task4.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data.clean import clean, load_raw

RAW = ROOT / "data" / "raw" / "investments_VC.csv"
OUT_CSV = ROOT / "data" / "processed" / "startups_clean.csv"
OUT_DIR = ROOT / "Task4" / "outputs"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    raw = load_raw(RAW)

    # ---- inspection of the raw file --------------------------------------------------
    inspect = pd.DataFrame({
        "dtype": raw.dtypes.astype(str),
        "missing_pct": (raw.isna().mean() * 100).round(1),
        "n_unique": raw.nunique(),
        "example": [raw[c].dropna().iloc[0] if raw[c].notna().any() else None for c in raw.columns],
    })
    inspect.to_csv(OUT_DIR / "raw_column_inspection.csv")

    df, log = clean(raw)
    df.to_csv(OUT_CSV, index=False)

    # ---- outlier review (funding): flag, inspect, keep ------------------------------
    f = df.funding_total_usd.dropna()
    lf = np.log10(f)
    q1, q3 = lf.quantile([.25, .75])
    hi, lo = q3 + 1.5 * (q3 - q1), q1 - 1.5 * (q3 - q1)
    log["funding_outliers_log10_iqr_high"] = int((lf > hi).sum())
    log["funding_outliers_log10_iqr_low"] = int((lf < lo).sum())
    log["funding_iqr_high_threshold_usd"] = float(10 ** hi)
    top = df.nlargest(15, "funding_total_usd")[["name", "market", "status", "funding_total_usd", "funding_rounds"]]
    top.to_csv(OUT_DIR / "top_funding_review.csv", index=False)
    df.nsmallest(10, "funding_total_usd")[["name", "funding_total_usd"]].to_csv(
        OUT_DIR / "bottom_funding_review.csv", index=False)

    # ---- skew before/after log, standardisation demo ----------------------------------
    skew = pd.DataFrame({
        "skew_raw": df[["funding_total_usd", "seed", "venture", "funding_rounds"]].astype(float).skew(),
        "skew_log1p": np.log1p(df[["funding_total_usd", "seed", "venture", "funding_rounds"]].astype(float)).skew(),
    })
    skew.round(2).to_csv(OUT_DIR / "skew_before_after_log.csv")

    with open(OUT_DIR / "cleaning_log.json", "w") as fh:
        json.dump(log, fh, indent=2, default=str)
    print(json.dumps(log, indent=2, default=str))
    print(skew.round(2))

    from Task4.eda import run_eda  # noqa: E402
    run_eda(df, OUT_DIR)


if __name__ == "__main__":
    main()
