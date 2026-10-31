# Data card: Olist Brazilian E-Commerce

| Item | Value |
| --- | --- |
| Used for | Market share (M7), competitor-move classifier (M7), peer cohorts (M6), review text (M4, stretch) |
| Source | Hugging Face mirror `miminmoons/olist-ecommerce-for-delivery-and-review-prediction`, raw tables copied from the public Olist dataset (Kaggle, 2018) |
| Licence | **CC BY-NC-SA 4.0** (original). The mirror's MIT label is invalid. Non-commercial, attribute, share-alike |
| Raw files | `data/raw/olist_hf/`: orders, order_items, products, sellers, order_reviews, category translation |
| Unit of a row | One order item (processed items); one seller in one category in one month (market table) |
| Coverage | Brazilian marketplace, orders 2016-09 to 2018-10; used window 2017-01 to 2018-08 |

## Why this dataset

Every seller's sales in a category are visible, so market share and competitor moves can be measured, not guessed. This is the only dataset here where competitor predictions can be checked against truth.

## Cleaning (`python -m startupiq.clean olist`)

Items joined to orders and products (1,603 items without category labelled "unknown"); 549 items in cancelled, unavailable or created orders removed; 349 items outside 2017-01 to 2018-08 removed; 814 duplicate review IDs removed. Details: `docs/results/cleaning_log.md`.

## Processed outputs

- `olist_items.parquet`: 111,752 items with category and month.
- `olist_seller_category_month.parquet`: 24,547 rows (3,029 sellers, 74 categories) with revenue, items, orders, median price, distinct products, share of category revenue.
- `olist_category_month.parquet`: category revenue, active sellers, HHI per month. 771 of 1,246 category-months have 5+ sellers.
- `olist_reviews.parquet`, `olist_sellers.parquet`.

## Known limits

- One country, one marketplace, two years. Sellers who left early are under-represented (survivorship).
- Very concentrated: Gini 0.79 on seller revenue. Most sellers are tiny and sparse.
- Competitor-move labels are not built here; they are defined in Step 4 (devdocs decision 0006).
- Review text is Portuguese.
