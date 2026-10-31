# Leakage checklist (Step 4)

Leakage = a model sees information at training or prediction time that it would not have when used for real. Every item below has a test or a code reference. Signed off on 2026-10-24.

## Rossmann daily forecaster (`features/rossmann.py`)

| # | Check | Status | How it is enforced |
| --- | --- | --- | --- |
| R1 | Customers (known only after the day) is not an input | Pass | Not in `FEATURES`; `test_rossmann_never_uses_customers` |
| R2 | No sales from the forecast window feed any feature | Pass | `test_rossmann_features_ignore_sales_after_origin` multiplies post-origin sales by 10 and checks every feature is unchanged |
| R3 | Lagged sales are at least one horizon old | Pass | `lag_364` and `roll_42_28` (window ends 42 days back) |
| R4 | Store averages use the training window only | Pass | `_encodings(d, origin)` filters `Date < origin` |
| R5 | Planned information only: promotions, holidays, opening schedule | Pass (assumption) | In the Kaggle setup these are given for the test period; a founder also knows their promo plan and holidays. Stated in decision 0005 |
| R6 | Validation chooses settings; test scored once | Pass | `--final` flag; validation-only runs never build the test features |
| R7 | In-sample target encoding for training rows | Accepted risk | Store means include the training row itself (about 1 of 800 rows per store); effect negligible, noted |

## Weekly global forecaster (`features/weekly.py`)

| # | Check | Status | How it is enforced |
| --- | --- | --- | --- |
| W1 | Features use only `y[:origin]` | Pass | `test_weekly_features_ignore_values_from_origin_on` |
| W2 | Training targets end before the evaluation origin | Pass | `_training_rows` stops at `origin - 12` per series |
| W3 | Scaling uses the pre-origin level | Pass | `scale_of(y, o)` = mean of the 12 weeks before `o` |
| W4 | Seasonal lag is never from the forecast window | Pass | `seasonal_same_week` uses `t - 52 < o`, else NaN |

## Olist competitor moves and market share (`features/olist.py`)

| # | Check | Status | How it is enforced |
| --- | --- | --- | --- |
| O1 | Features use months up to t | Pass | Lags are `shift(+k)`; label inputs are `shift(-k)` and only feed `label` / `share_next` |
| O2 | Reviews used only from orders bought up to t-1 | Pass | Cumulative sums minus the current month |
| O3 | Labels look at t+1..t+3 only, and no label crosses its split's end | Pass | Exit = no orders in t+1..t+3 (changed from "never again", which looked to the end of the data); split boundaries in `splits.py`; `test_olist_split_boundaries` |
| O4 | Row eligibility uses the past only | Pass | Rows need orders at t and 2+ active months up to t; the exploration filter "active 6+ months overall" used future months and was dropped |
| O5 | Category totals, shares, HHI at t use month t only | Pass | Grouped by (category, month) |
| O6 | Seller identity can repeat across splits | Accepted | The split is temporal; a seller seen in training appears later, which is the real use (we know a rival's history) |

## Trend regime (`models/regime.py`)

| # | Check | Status | How it is enforced |
| --- | --- | --- | --- |
| G1 | Features from `y[:origin]` only; "same weeks last year" uses `y[o-52:o-40]` | Pass | `_row` |
| G2 | Training labels end before the validation origin | Pass | `o + 12 <= val_origin` |

## Clustering (`models/clustering.py`)

| # | Check | Status | How it is enforced |
| --- | --- | --- | --- |
| C1 | Store profiles from the training window only | Pass | Filters `< ROSS_TEST_START` |
