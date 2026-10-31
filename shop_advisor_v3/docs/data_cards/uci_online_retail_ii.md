# Data card: UCI Online Retail II

| Item | Value |
| --- | --- |
| Used for | Forecaster generalisation test (a different business); messy-input tests for intake (M0); product-level series |
| Source | UCI Machine Learning Repository, dataset 502, `online_retail_II.xlsx` |
| Licence | CC BY 4.0 per the UCI listing; cite Chen (2012) and UCI |
| Raw file | `data/raw/uci_retail/online_retail_ii.zip` (two sheets, 2009-2010 and 2010-2011) |
| Unit of a row | One invoice line (raw); one product-week or one week (processed) |
| Coverage | UK online gift wholesaler, 2009-12-01 to 2011-12-09, 43 countries (UK 85% of revenue) |

## Cleaning (`python -m startupiq.clean uci`)

| Rule | Lines removed |
| --- | ---: |
| Exact duplicate lines | 34,335 |
| Non-product codes (POST, DOT, M, C2, D, S, ...) | 5,980 |
| Cancellations (17,914) netted off against the latest matching sale (6,854) | 24,768 |
| Remaining zero or negative quantity or price | 5,923 |

226,637 lines without a customer ID are kept (customer is not needed for sales). Unmatched cancellations are kept in a separate returns table. Weeks with no trading (Christmas closures) are filled as zero. 1,067,371 raw lines become 996,365 clean lines. Details: `docs/results/cleaning_log.md`.

## Processed outputs

- `uci_lines.parquet` (clean lines), `uci_returns.parquet` (cancellation lines)
- `uci_total_weekly.parquet`: 104 complete weeks, 2009-12-13 to 2011-12-04 (2 zero weeks)
- `uci_product_weekly.parquet`: 192,532 product-weeks for 4,836 products

## Known limits

- One retailer: no competitor information.
- Strong Christmas wholesale peak (October to November weeks 1.89x January to September).
- 43% of products sell in fewer than 26 weeks; product-level forecasting is only sensible for the 1,604 products with 52+ active weeks.
- Cancellation matching is by customer, product and quantity; partial returns that do not match are not subtracted.
