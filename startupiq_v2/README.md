# Startup IQ v2

A startup intelligence platform for founders, investors, and incubators.

**First capability: Peer Intelligence.** Paste a company description and get its likely sector, its closest peers, its place on a market map, and the range of what comparable companies have raised. Every output links to its evidence, and model output is labelled as inferred.

## Status

Planning. No code yet. See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

## Principles

- Real data only for reported results. Synthetic data is allowed for tests and demos and must be labelled.
- No startup-success prediction. Published leakage-controlled results are weak and the best data is licensed.
- Every model beats a simple baseline on the same split, or it stays an experiment.
- Evidence first: outputs trace to source records.

## Data streams

| Stream | Source | Used for |
| --- | --- | --- |
| A, text | Y Combinator company directory (unofficial `yc-oss/api`) | Sector classification, peer retrieval, clustering, deep text model |
| B, structured | SEC Form D data sets | Raise-size benchmark |

Licence status of both sources must be confirmed before any commercial use. See the plan, section 7.
