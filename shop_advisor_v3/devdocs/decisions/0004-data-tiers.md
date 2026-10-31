# 0004: Data-quality tiers and refusal thresholds

- Status: Accepted (2026-10-24) on the Step 5 tier check; see the evidence section
- Date: 2026-10-24
- Module / task: M0, M10; Tasks 4, 10

## Problem

Uploads vary from 3 weeks to 3 years. The system must not produce confident output from thin data.

## Current thresholds (implemented in `src/startupiq/intake/series.py`)

Monthly uploads count 52/12 weeks per month. More than half of periods empty caps the tier at Basic.

| Weekly history | Tier | Allowed |
| --- | --- | --- |
| Under 20 usable rows (daily or transaction uploads), under 4 periods with sales, or all zeros | Refused | Message explaining what is needed, plus the template |
| Under 8 weeks | Summary only | Totals, simple trend; no forecast |
| 8 to 25 weeks | Basic | Baseline forecast with wide band; no seasonal claims |
| 26 to 51 weeks | Standard | Better forecast, weekly pattern, stock and cash rules |
| 52+ weeks | Full | Yearly seasonality, neural forecaster, what-if |

## Why these numbers

- 8 weeks: two 4-week windows, the minimum for the growth calculation in vitals.
- 52 weeks: a seasonal model needs at least one full year to see each season once.
- Middle cut at 26 weeks: half a year, enough for a reliable weekly-pattern estimate.

## How we will check them (Step 5)

Truncate Rossmann and UCI series to 8, 13, 26, 39, 52 weeks and measure forecast error vs history length. If error at a tier is no better than the store mean, move the threshold up. Result goes in the evaluation log and this record moves to Accepted.

## Evidence (Step 5 tier check, Rossmann stores, MASE, validation origin)

| Weeks of history | Moving average | ETS | Seasonal naive | Seasonal, level-adjusted |
| --- | --- | --- | --- | --- |
| 8 | 0.897 | n/a | n/a | n/a |
| 26 | 0.897 | 0.940 | n/a | n/a |
| 52 | 0.897 | 0.928 | 0.492 | n/a |
| all (about 120) | 0.897 | 0.906 | 0.492 | 0.418 |

Smoothing methods are as good at 8 weeks as with full history, so a basic forecast from 8 weeks is justified. Seasonal methods need a full year and then roughly halve the error, so seasonal forecasts stay in the Full tier (52+ weeks). The 26-week Standard boundary is not separated by this test (no method gains between 8 and 51 weeks); it is kept because the weekday pattern and the stock and cash rules need half a year of history to be stable, an argument still to be tested in Step 7.
