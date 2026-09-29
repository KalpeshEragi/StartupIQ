# Task 4 — Cleaning decisions

| Issue found | Decision | Why |
|---|---|---|
| 4,856 fully blank rows (these are the "4,855 duplicates") | Dropped | No information |
| Duplicate permalinks (2 companies, 4 rows) | Kept the more complete row | Same company recorded twice |
| Column names with leading spaces (` market`, ` funding_total_usd`) | Stripped | Consistency |
| `funding_total_usd` is text with Indian digit grouping (`17,50,000`) and `-` | Commas removed, converted to number; `-` → missing (8,531 rows) | `-` means unknown, not zero |
| Funding-type columns are 0 where total is `-` | Reset to missing in those 8,531 rows | Total equals the sum of the type columns exactly everywhere it is known, and every company has ≥1 round, so those zeros mean "unknown" |
| Unparsable dates (years 0001–0201, 1636) — 1 + 10 + 6 values | Set to missing | Impossible values |
| 67 companies founded before 1900; 493 with first funding >1 year before founding | **Kept**, flagged (`flag_founded_before_1900`, `flag_funding_before_founding`) | Possibly wrong or legitimate; Task 5 decides how to use them |
| Funding outliers (14 above the log-IQR fence, up to $30B) | **Kept** | Reviewed the top 15: Verizon, Facebook, Flipkart, Uber, Solyndra, etc. Real large values, though many are not startups. Handled with log transform |
| Very small values ($1–$100) | Kept | Plausible micro-funding; log transform handles them |
| Heavy skew | Added `funding_total_usd_log1p` | Skew 142 → -0.27 |
| Scaling | Not applied to the saved dataset | Scalers must be fitted on the training split only (Task 5) |
| Missing state / category / dates | Left missing | Not invented; handled in modelling pipeline |
| `status` missing (1,314) | Left missing | Excluded from supervised target in Task 6 |
