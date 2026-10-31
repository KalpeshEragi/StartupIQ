"""Train the TCN that the product serves (data/models/dl_weekly_tcn_serve.pt).

python -m startupiq.dl.serve_train

Trained on every window of the public weekly panel (Rossmann stores, UCI products) whose targets
end by the last week of the data, with the settings chosen in docs/results/dl_weekly.json. The
product backtests it on the founder's own last 12 weeks before using it (models/forecast_baseline.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from startupiq.dl import windows as W
from startupiq.dl.train import Config, fit
from startupiq.features import weekly as fw

OUT = Path("data/models/dl_weekly_tcn_serve.pt")
CONFIG = Path("docs/results/dl_weekly.json")


def main() -> None:
    series, weeks = fw.panel()
    cfg = Config(**json.loads(CONFIG.read_text(encoding="utf-8"))["chosen"]["tcn"]["config"])
    train = W.build(series, weeks, W.training_origins(series, {s: len(y) for s, y in series.items()}, step=2))
    print(f"{len(train)} windows from {len(series)} series", flush=True)
    t = fit(train, cfg, seed=0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t.model.state_dict(), OUT)
    print(f"saved {OUT} after {t.epochs} epochs in {t.seconds:.0f}s; last (train, holdout) loss {t.history[-1]}", flush=True)


if __name__ == "__main__":
    main()
