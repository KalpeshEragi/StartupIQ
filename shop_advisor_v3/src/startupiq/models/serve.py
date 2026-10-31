"""Fitted Step 5 models for use in the product (decision 0012).

The competitor-move random forest and the share LightGBM are refit on the train and
validation months with the settings chosen in Step 5, then cached in data/models/.
They score rows in the Olist feature format (features/olist.py): the "marketplace mode",
where a seller and its rivals are all visible.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from startupiq.features import olist as fo

CACHE = Path(__file__).resolve().parents[3] / "data/models"


@lru_cache(maxsize=1)
def olist_rows() -> pd.DataFrame:
    return fo.build()


def _fit_or_load(name: str, fit):
    path = CACHE / f"{name}.joblib"
    if path.exists():
        return joblib.load(path)
    model = fit()
    CACHE.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model


def competitor_model():
    from startupiq.models.competitor import make_models
    rows = olist_rows()
    tr = rows[rows.split.isin(["train", "val"])]
    return _fit_or_load("competitor_rf", lambda: make_models(0)["Random forest"].fit(tr[fo.FEATURES], tr.label))


def share_model():
    from startupiq.models.share import MIN_SELLERS, _models
    rows = olist_rows()
    tr = rows[rows.split.isin(["train", "val"]) & (rows.category_sellers >= MIN_SELLERS) & rows.share_next.notna()]
    return _fit_or_load("share_lgbm", lambda: _models(0)["LightGBM (L1 loss)"].fit(tr[fo.FEATURES], tr.share_next))


def move_probabilities(rows: pd.DataFrame) -> pd.DataFrame:
    m = competitor_model()
    proba = m.predict_proba(rows[fo.FEATURES])
    return pd.DataFrame(proba, columns=[f"p_{c}" for c in m.classes_], index=rows.index)


def next_share(rows: pd.DataFrame) -> np.ndarray:
    return np.clip(share_model().predict(rows[fo.FEATURES]), 0, 1)


def train_weekly_tcn() -> Path:
    """Train the serving TCN (chosen settings, seed 0) on every origin of the whole weekly panel.
    python -c "from startupiq.models.serve import train_weekly_tcn; train_weekly_tcn()"  (GPU recommended)"""
    import json

    import torch

    from startupiq.dl import windows as W
    from startupiq.dl.train import Config, fit
    from startupiq.features import weekly as fw
    from startupiq.models.forecast_baseline import TCN_FILE
    series, weeks = fw.panel()
    cfg = Config(**json.loads((Path(__file__).resolve().parents[3] / "docs/results/dl_weekly.json").read_text(encoding="utf-8"))["chosen"]["tcn"]["config"])
    train = W.build(series, weeks, W.training_origins(series, {s: len(y) for s, y in series.items()}))
    t = fit(train, cfg, 0)
    TCN_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t.model.state_dict(), TCN_FILE)
    return TCN_FILE
