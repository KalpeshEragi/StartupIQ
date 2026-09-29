# Task 5 — Feature engineering and leakage analysis

Full list with source columns and formulas: `Task5/outputs/feature_registry.csv` (28 candidate features).

## Key engineered features
- `age_at_first_funding` = (first funding − founded)/365.25. Values below −1 year (493 companies) are treated as invalid → missing; −1..0 → 0; capped at 40. Why: measures how fast a company raised money, using only information at first funding.
- `n_categories` = number of tags in `category_list` (breadth of description).
- `has_seed`, `has_angel`, `has_convertible_note`, `has_grant`, `has_crowdfunding`: presence flags that stay missing (not 0) where funding is unknown.
- Full-history only: `log_funding_total`, `funding_span_years`, `rounds_per_year` (velocity), `years_since_last_funding`, `n_named_rounds`/`max_named_round` (stage proxy), `venture_share`, `avg_funding_per_round_log`.
- Not built: text/category tag embeddings, investor features (not in this dataset), city (too high-cardinality; `region` is used as hub proxy).

## Leakage decisions
Question asked per feature: would StartupIQ know this when evaluating an early-stage startup?

| Feature group | Verdict | Reason |
|---|---|---|
| Market, country, region, tag count | Safe | Static description (tags are as of the snapshot; small residual risk) |
| Age at first funding | Safe | Uses first round only |
| Seed / angel / convertible / grant / crowdfunding | Safe with caveat | Early instruments, but amounts are cumulative over all rounds of that type |
| Total funding, round count, venture, private equity, debt, post-IPO, A–H rounds | **Future information** | Include later rounds; companies that get acquired keep raising, so these partly encode the outcome |
| `last_funding_at`, `years_since_last_funding`, `funding_span_years`, `rounds_per_year` | **Future information** | Companies that closed or were acquired stop raising; the date reflects the outcome |
| `founded_year`, `first_funding_year` | Calendar effect | Acquired share falls from ~85% (1990s cohorts) to ~45% (2010s); this reflects the observation window, and a new startup is outside the range. Excluded from primary set |
| `homepage_url`, `name`, `permalink`, city | Not used | Identifier/high-cardinality; homepage presence is recorded at snapshot time |
