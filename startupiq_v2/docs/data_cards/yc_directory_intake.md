# YC directory: Phase 1 intake report

Source: `https://yc-oss.github.io/api/companies/all.json` (unofficial; no licence stated). Retrieved 2026-10-13. SHA-256 and size are in `data/raw/MANIFEST.md`. All numbers below were computed from the downloaded file.

## Shape

| Item | Proposal said | Found |
| --- | --- | --- |
| Companies | 6,201 (repo page) | **6,288** records, no duplicate ids |
| Fields per record | about 37 (repo page); 28 in one batch file | **29**, present in every record |
| Industries | not stated | **9** |
| Subindustries | 59 industries (repo page wording) | **59** distinct subindustry values |
| Tags | 336 (repo page) | **338** distinct |
| Batches | 50 | **51** |
| Status values | Active, Acquired, Inactive | Active 4,344; Inactive 1,089; Acquired 832; **Public 23** |
| Stage values | Early, Growth | Early 5,190; Growth 1,098 |

## Class balance (the target for sector classification)

Industry: B2B 3,202 (51%), Consumer 887, Healthcare 703, Fintech 663, Industrials 484, Real Estate and Construction 163, Education 125, Government 43, Unspecified 18.

Consequences: accuracy alone would be misleading (always predicting B2B scores about 51%), so macro-F1 is the primary metric. "Unspecified" (18) is treated as unlabelled, not as a class. Government (43) is small and may need merging or extra care. Of 59 subindustries, 2 have fewer than 20 companies; 67 tags have at least 50 companies.

## Missing values

| Field | Empty or null | Share |
| --- | --- | --- |
| long_description | 416 | 6.6% |
| one_liner | 159 | 2.5% |
| both texts empty | **53** | 0.8% (no usable text; exclude from text models) |
| tags | 863 | 13.7% |
| team_size null | 110 | 1.7% |
| team_size equals 0 | 136 | 2.2% (treat as missing) |
| all_locations | 152 | 2.4% |
| website | 39 | 0.6% |
| former_names | 3,213 | 51% (normal; most companies never renamed) |
| tags_highlighted, app_answers | about 98.5% | not useful; drop |

## Duplicates

- Ids: 0 duplicates.
- **Names: 104 repeated, but spot checks show different companies sharing a name** (for example "Tilt" 2012 and 2021 with different websites; "Apollo" three batches, three websites). Do not deduplicate by name.
- Websites: 43 repeated. One-liners: 167 repeated. These are the candidates for near-duplicate review.

## Leakage confirmation

Five tags are literally industry names: B2B, Healthcare, Fintech, Education, Consumer. Using tags as inputs to predict industry would leak the label. Confirmed as an exclusion in the plan.

## Text size

One-liner median 7 words (max 16). Long description median 54 words, 95th percentile 183, maximum 1,180. Short texts, as expected.

## Decisions taken from this report

1. Primary metric for sector classification is macro-F1.
2. Exclude the 53 records with no text from text models; keep them in the log.
3. Treat team_size 0 as missing.
4. Drop tags_highlighted and app_answers.
5. Dedupe on id, then website, then near-duplicate text; never on name alone.
6. Keep status as a descriptive field only, never as a target.

## Still open

Licence of the YC data. Terms of use on the YC website. Whether the 9-industry taxonomy is stable across batches (check in Phase 2).
