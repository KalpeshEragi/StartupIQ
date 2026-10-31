# Datasets evaluated

All numbers below were measured on 2026-10-23/24 by the scripts in `scripts/exploration/`. Raw files are in `data/raw/` and listed with SHA-256 in `data/raw/MANIFEST.md`.

## Rossmann Store Sales (forecasting)

- Source: Hugging Face mirror `AiiN-aini/rossmann-store-sales` (copy of the Kaggle competition files `train.csv`, `store.csv`).
- 1,017,209 store-days, 1,115 stores, 2013-01-01 to 2015-07-31; 758 to 942 days per store. No missing values or duplicate (store, date) pairs in `train.csv`.
- 17% of rows are closed days (`Open = 0`); 54 rows open with zero sales. Promo days sell 1.39x non-promo days.
- `store.csv`: CompetitionDistance missing for 3 stores; competition-open dates missing for 354; Promo2 fields missing for 544 (stores not in Promo2).
- Stores with a rival closer than the median distance (2,325 m) average 7,148 vs 6,766: competition distance alone is a weak signal.
- Kaggle test labels are not included, so we cannot compare with the leaderboard.

## UCI Online Retail II (generalisation, intake tests)

- Source: UCI archive, `online_retail_II.xlsx`, two sheets.
- 1,067,371 lines, 5,305 products, 5,942 customers, 43 countries, 2009-12-01 to 2011-12-09.
- Problems (good for Task 4): 2.2% negative quantities, 1.8% cancellation invoices, 0.6% zero or negative price, 3.2% duplicate lines, 22.8% missing customer ID.
- 1,648 products have 52+ active weeks; top-200 products have zero sales in 17.8% of weeks; median weekly coefficient of variation 1.16 (noisy).
- Single retailer: no competitor information.

## Olist Brazilian E-Commerce (competitors, market share)

- Source: Hugging Face mirror `miminmoons/olist-ecommerce-for-delivery-and-review-prediction`, raw tables in `data/`.
- 99,441 orders, 112,650 items, 3,095 sellers, 32,951 products, 2016-09 to 2018-10. 99,224 reviews, 41% with text (Portuguese).
- Usable window 2017-01 to 2018-08 (20 months). 751 category-months have 5+ active sellers; 40 of 73 categories have a median of 5+ sellers per month, 18 have 20+.
- Concentration: top 1% of sellers make 26% of revenue, top 10% make 67%; 58% of sellers have fewer than 10 orders.
- 1,361 seller-category pairs are active 6+ months, 651 are active 10+.
- Draft move labels (12,790 seller-category-months): stagnate 39%, price cut 25%, no orders next month 21%, expansion 15%. Known flaws: price cut uses the median price of whatever sold (product-mix noise); "no orders" is mostly a sparse month, not an exit. Both are redefined in Step 4 (decision 0006).

## FRED and Wikipedia (market trend)

- FRED: RSXFS (retail ex food services), RSCCAS (clothing stores), MRTSSM4541USN (electronic shopping) monthly from 1992, about 417 rows each.
- Wikipedia monthly page views from 2015-07: Coffee and Smartphone 111 months; Wireless_earbuds only 25 months (new article). Page views are a proxy for interest, not sales.

## After cleaning (Step 2)

Rule-by-rule counts: `docs/results/cleaning_log.md`. Reasons for each rule: decision [0010](decisions/0010-cleaning-rules.md). Reader-facing summaries: `docs/data_cards/`.

| Dataset | Clean output | Notes for modelling |
| --- | --- | --- |
| Rossmann | 1,017,209 store-days (flags, nothing dropped); 143,435 complete store-weeks | 54 missing sales, 482 outlier flags, 180 gap stores; exclude Customers from inputs |
| UCI | 996,365 lines from 1,067,371; 104 complete weeks; 4,836 products | Largest remaining line is 19,152 units: check in Step 4 whether bulk orders need a cap |
| Olist | 111,752 items; 24,547 seller-category-months; 771 category-months with 5+ sellers | Move labels built in Step 4 |
| Signals | One monthly long table | Wikipedia trailing partial month removed |

EDA findings that change modelling choices (from `docs/results/eda_summary.md`):

- Rossmann promotions alternate weekly (promo weeks 1.36x non-promo weeks) and December is 1.26x other months: promo and calendar inputs are mandatory.
- Customers correlates 0.82 with sales but is unknown in advance: excluded (leakage).
- Competitor distance barely relates to sales (Spearman -0.03): it will not carry the competitor story; Olist does.
- UCI October-November weeks are 1.89x January-September; 43% of products sell in under 26 weeks.
- Olist orders grow 8.2x over the window, so seller growth must be read as share of category; seller revenue Gini 0.79.

## Licences

See decision [0002](decisions/0002-datasets-and-licences.md). Short version: lab and academic use only; Olist is CC BY-NC-SA 4.0 even though the mirror says MIT.

## Not used

YC directory and SEC Form D (previous project direction) stay in `data/raw/` but feed nothing.
