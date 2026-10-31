# 0010: Cleaning rules

- Status: Accepted
- Date: 2026-10-24
- Module / task: Task 4; `src/startupiq/clean/`

## Principles

1. **Flag before you drop.** A row is dropped only when it is certainly not a sale (duplicate, fee, cancelled order). Anything that might be real (outliers, gaps) is flagged so each model can decide.
2. **Never fill what the source did not record.** Gaps stay gaps. Filling the Rossmann 2014 gap with interpolated sales would teach the model invented data.
3. **Impute only inside the training pipeline.** Values like missing competition distance are left NaN here and imputed from training data only (Step 4), to avoid test-set leakage.
4. **Every rule logs rows before and after.** One command rebuilds everything: `python -m startupiq.clean`.

## Rule-by-rule choices

| Dataset | Problem | Choice | Alternative rejected and why |
| --- | --- | --- | --- |
| Rossmann | Open day with zero sales (54) | Set sales to missing, keep the row | Keep as zero: would teach the model that open days can have no demand. Drop: loses the calendar row |
| Rossmann | Extreme days (482 with robust z > 5) | Flag only | Drop or clip: many are real (pre-holiday peaks); robust z on log scale with median and MAD resists the outliers it is looking for, unlike a mean and standard-deviation z-score |
| Rossmann | 180 stores missing 2014-07 to 2014-12 | Flag the stores, leave the gap | Interpolate: invents six months of data. Drop the stores: loses 16% of series |
| Rossmann | Promo2 fields blank for stores not in Promo2 | Set to 0 / "none" | Treat as missing: they are structurally empty, not unknown |
| Rossmann | Customers column | Kept in data, excluded from inputs in Step 4 | It is not known in advance; correlation 0.82 with sales would leak |
| UCI | Postage, fees, adjustments (codes not starting with 5 digits) | Drop | Keep: they are not product demand |
| UCI | Cancellations | Remove the cancellation and the latest earlier sale with the same customer, product and quantity | Subtract all cancellations from demand: partial returns would reduce sales that did happen. Ignore cancellations: overstates demand (the famous 80,995-unit order was cancelled) |
| UCI | Zero or negative remaining lines | Drop | They are damages and write-offs, not sales |
| UCI | Missing customer ID (23%) | Keep, flag | Drop: would remove a quarter of real sales; customer is not needed for forecasting |
| UCI | Christmas closure weeks | Zero-filled weekly calendar | Leave missing: the business was closed, so zero is the true value |
| Olist | Cancelled, unavailable, created orders | Drop | They never became sales |
| Olist | 2016 and after 2018-08 | Outside window | 2016 has under 1,000 orders a month; the end is truncated mid-month |
| Olist | Duplicate review IDs (814) | Keep first | The same review attached to several orders of one customer |

## Consequences

Models must handle NaN sales and the outlier and gap flags; LightGBM does natively, the neural network needs a mask (Step 8).
