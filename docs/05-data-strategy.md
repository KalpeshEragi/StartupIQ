# 05 — Data Strategy

## Philosophy
Do not force one dataset to power the entire product. Different real-world sources serve different modules.

## Source Categories
- Startup/company history
- Funding/investment
- Investor data
- Incubator data
- Market data
- Documents/websites
- Live external intelligence

## Rules
- Integrate sources logically; do not blindly merge incompatible datasets.
- Synthetic data may be used for controlled testing/demonstration only, and must never be presented as real-world evidence.
- Unknown startup information is never invented; it stays unknown.
- Useful research evidence is stored and reused (see `07-automation-and-agents.md`).

## Decided (first ML pipeline)
Primary ML dataset: Crunchbase "StartUp Investments" (`investments_VC.csv`, Kaggle `arindam235`), 54,294 raw rows → 49,436 companies after cleaning. Target: acquired vs closed among resolved outcomes. See `Task4/`–`Task6/`. Investor/incubator data and storage remain undecided (see `09-open-decisions.md`).
