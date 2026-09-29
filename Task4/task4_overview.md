# Task 4 — Data Understanding, EDA, Preprocessing

**What:** load `data/raw/investments_VC.csv`, explore it, clean it, and save a reproducible cleaned dataset.
**Run:** `python Task4/run_task4.py` (rebuilds everything from raw; raw is never modified).

| Output | Location |
|---|---|
| Cleaned dataset (49,436 × 42) | `data/processed/startups_clean.csv` |
| Cleaning code | `src/data/clean.py` |
| EDA code / figures / tables | `Task4/eda.py`, `Task4/outputs/` |
| Cleaning counts | `Task4/outputs/cleaning_log.json` |

**Raw file:** 54,294 rows × 39 columns, one row per company (Crunchbase snapshot, funded companies, up to ~Dec 2014).
**Result:** 4,856 blank rows removed; 49,436 real companies remain. Every company has at least one funding round.
Details: `task4_eda.md`, `task4_preprocessing.md`.
