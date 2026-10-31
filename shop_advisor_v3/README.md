# Startup IQ

A weekly business checkup for small sellers. Upload a sales sheet; get a forecast with a range, stock and cash warnings, market direction, likely competitor moves, estimated market share, and a recommended plan, each with its reason shown.

Built for the Computational Intelligence Lab (CELBC710): rules and search (Tasks 1 to 3), classical ML (Tasks 4 to 6), deep learning (Tasks 7 and 8), integration and validation (Tasks 9 and 10).

## Status

- Step 0 done: package skeleton, data contracts, intake v0, vitals, stub pipeline returning all cards end to end.
- Steps 1 and 2 done: data cards, scripted cleaning of Rossmann, UCI Online Retail II, Olist and market signals with before/after logs, 15 EDA figures with findings.
- Step 3 done: intake reads messy CSV/Excel uploads (title rows, ₹ amounts, no headers, Hindi headers, wide sheets, several sheets, US or Indian dates), asks when unsure, and grades the data. Tested on a 25-file broken-file pack and real exports.
- Steps 4 and 5 done: leak-free features and splits, competitor-move labels, six classical experiments scored on validation and then once on test. The forecast card now uses the validated Auto method (pooled MASE 0.616 vs 0.751 for a moving average). Choices and evidence: `devdocs/decisions/0012-step5-model-choices.md`.
- Steps 6 and 7 done: knowledge model (frames, semantic network, constraint checker) and a 47-rule engine with forward and backward chaining and certainty factors. Rules turn the forecast and Step 5 model outputs into warnings, diagnoses, a pre-mortem and recommendations, and a safety gate blocks moves that break a constraint. Evidence: `devdocs/decisions/0007`, `0013`.
- Step 10 done: planner searching over plans of business moves with BFS, DFS, uniform-cost, A* and greedy; A* finds the optimal plan in every benchmark case and its plan appears on the next-move card. Evidence: `devdocs/decisions/0014-planner.md`.
- Steps 8, 9, 11 and 12 done: deep forecasters (MLP, LSTM with attention, TCN) on the GPU, with the hybrid Auto + TCN serving the forecast card; interpretability study; FastAPI web app; validation on 300 unseen sellers; ethics, report and slide outline. Start here: `docs/REPORT.md`.

See [docs/PLAN_TASKS_1_TO_10.md](docs/PLAN_TASKS_1_TO_10.md) for the 13 steps.

## Run

```bash
pip install -e ".[dl,api,dev]"
python -m pytest -q
python -m startupiq.clean
python -m startupiq.eda
uvicorn startupiq.api.app:app   # then open http://127.0.0.1:8000
```

Cleaning reads `data/raw/` (downloads listed in `data/raw/MANIFEST.md`); the first run reads the UCI workbook in about 2 minutes, then caches it.

Deep models (Steps 8-9) train on an NVIDIA GPU when PyTorch's CUDA build is installed; on CPU they work but are roughly 25x slower. The project was run with PyTorch 2.9.1 for CUDA 12.8 on an RTX 2050 (4 GB):

```bash
pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128
```

## Where things are

| Folder | For |
| --- | --- |
| `docs/` | Plan and data cards (readers of the project) |
| `devdocs/` | Architecture, evaluation protocol and log, and a decision record for every technique choice (builders and examiners) |
| `src/startupiq/` | Code; entry point `pipeline.analyze()` |
| `scripts/exploration/` | Scripts that produced the dataset numbers in `devdocs/` |
| `data/raw/` | Downloads (git-ignored) and `MANIFEST.md` |

## Principles

- Forecasts are ranges. Competitor and share outputs are labelled as inferred.
- Every model beats a stated baseline on the same split, or it is not used.
- The system refuses to answer when the data is too thin.
- Datasets are for lab and academic use only (see `devdocs/decisions/0002-datasets-and-licences.md`).
