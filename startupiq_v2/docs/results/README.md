# Results, Tasks 4-6 (real data only)

Reproduce: `PYTHONPATH=src python -m startupiq.pipelines.task4` then `task56_yc` and `task56_formd`.
Raw data is git-ignored; see data/raw/MANIFEST.md and src/startupiq/ingest/download.py.

| Task | Output | Headline |
| --- | --- | --- |
| 4 | task4/figures, cleaning logs | YC 6,288 rows; Form D 107,709 filings -> 32,206 operating-company originals |
| 5 | task5_6/figures/t5_* | tags/subindustry excluded as leakage; word TF-IDF + chi-squared top 5,000 |
| 6 YC | t6_model_comparison, t6_confusion_matrix, t6_market_map | test accuracy 0.80 (CI 0.78-0.82), macro-F1 0.70, top-3 0.98 vs majority 0.51 |
| 6 Form D | t6_formd_regression, t6_formd_bands | band accuracy 0.54 (majority 0.35), within one band 0.90; R2 0.25 |

Limits: Form D filing metadata carries little signal about raise size, so exact-band accuracy stays near 54%.
Clusters in YC text are weak (silhouette ~0.03), so the market map is a rough guide.
