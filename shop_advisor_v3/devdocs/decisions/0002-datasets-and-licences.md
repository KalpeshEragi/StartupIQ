# 0002: Datasets and licences

- Status: Accepted
- Date: 2026-10-24
- Module / task: all; Task 4, Task 10 (ethics and legal)

## Problem

We need (a) many sales series to train a forecaster, (b) a market where every competitor's sales are visible, so competitor and share predictions can be checked, and (c) a free demand signal. Kaggle is not set up on the build machine.

## Options considered

| Option | For | Against |
| --- | --- | --- |
| Rossmann via Hugging Face mirror | 1,115 parallel daily series, promo and competitor fields, clean | Unofficial copy, no licence on the mirror, original under Kaggle rules; no test labels |
| M5 / Walmart (Kaggle) | Huge, hierarchical | Needs Kaggle; heavy for a lab |
| UCI Online Retail II | Official source, CC BY 4.0, real mess | One retailer, no competitors, about 2 years |
| Olist via Hugging Face mirror | Every seller in each category observed; small businesses | Mirror mislabels licence as MIT; sparse per seller |
| Dominick's (Kilts) | Brand-level prices and promos, classic for share models | Not yet checked; large; grocery only |
| Google Trends | Popular demand signal | Unofficial API, rate limits |
| Wikipedia page views, FRED | Official, free, no key | Proxies only |

## Decision

Rossmann (forecast training), UCI Online Retail II (generalisation and intake tests), Olist (competitors and share), FRED and Wikipedia (market trend). Dominick's is the fallback if Olist proves too sparse.

## Licence position

| Dataset | Licence | What we may do |
| --- | --- | --- |
| Rossmann mirror | None stated on mirror; Kaggle competition rules apply to the original | Lab use; do not redistribute raw files; cite Kaggle |
| UCI Online Retail II | CC BY 4.0 (per UCI listing) | Use with attribution |
| Olist mirror | Original is CC BY-NC-SA 4.0; the mirror's MIT label is invalid | Non-commercial only, attribute, share-alike for derived data |
| FRED, Wikipedia | Public data; cite sources | Use with citation |

Raw files are git-ignored. A commercial version would need its own or licensed data.

## Consequences

Mirrors are unofficial, so we keep SHA-256 hashes in the manifest and note that row counts match the published descriptions. The report cites the original sources.
