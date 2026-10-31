# Data card: Rossmann Store Sales

| Item | Value |
| --- | --- |
| Used for | Training and benchmarking the sales forecaster (M3, Tasks 6 to 8); promotion effect for the planner (M9) |
| Source | Hugging Face mirror `AiiN-aini/rossmann-store-sales` of the Kaggle competition "Rossmann Store Sales" (2015) |
| Licence | None stated on the mirror; original under Kaggle competition rules. Lab use only, no redistribution, cite Kaggle. See devdocs decision 0002 |
| Raw files | `data/raw/rossmann_hf/train.csv` (1,017,209 rows), `store.csv` (1,115 rows); hashes in `data/raw/MANIFEST.md` |
| Unit of a row | One store on one day (raw); one store in one week (processed weekly) |
| Coverage | 1,115 German drugstores, 2013-01-01 to 2015-07-31 |

## Fields used

| Field | Meaning | Use |
| --- | --- | --- |
| Sales | Turnover for the day | Target |
| Open, Promo, StateHoliday, SchoolHoliday, DayOfWeek | Calendar and promotion | Inputs (known in advance) |
| StoreType, Assortment, CompetitionDistance, CompetitionOpenSince*, Promo2* | Store attributes | Static inputs |
| Customers | Customers that day | **Excluded from inputs**: not known in advance (correlation 0.82 with sales would leak) |

## Cleaning (`python -m startupiq.clean rossmann`)

No rows dropped from the daily table; problems are flagged so models can choose. 54 open days with zero sales set to missing; 482 store-days flagged as outliers (robust z > 5 on log sales); 180 stores flagged for the source's 2014-07-01 to 2014-12-31 gap (left empty, not filled); Promo2 blanks set to 0 for stores not in Promo2; 3 missing competition distances and 354 unknown competition-open dates flagged and left for train-only imputation. Full counts: `docs/results/cleaning_log.md`.

## Processed outputs

- `data/processed/rossmann_daily.parquet`: raw columns plus flags and store attributes.
- `data/processed/rossmann_weekly.parquet`: 143,435 complete store-weeks (2013-01-13 to 2015-07-26) with sales, open days, promo days, holiday days, missing days.

## Known limits

- German drugstores, not small online sellers: the forecaster must be tested on other data (leave-one-dataset-out).
- Kaggle test labels are absent; we hold out our own last 42 days, so scores are not comparable with the leaderboard.
- Promotions alternate weekly, which creates a strong saw-tooth (promo weeks 1.36x non-promo weeks). This is specific to Rossmann.
