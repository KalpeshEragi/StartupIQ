"""Product forecaster: the hybrid chosen in Step 8 (decision 0015).

Auto (decision 0012) picks, per upload, the method that did best on the upload's own most
recent periods. For weekly uploads with 64+ weeks the hybrid adds the TCN trained on the
public weekly panel as one more candidate: it is backtested on the same last 12 weeks and
used only if it did better there. The founder's data was never in the TCN's training set,
so that backtest is honest. If the TCN file is missing, plain Auto runs.
Validation: pooled MASE 0.603 (hybrid) vs 0.616 (Auto) on 2,539 series.
"""

from __future__ import annotations

from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from startupiq.contracts import Forecast, SalesSeries, Source
from startupiq.models.forecast_methods import METHODS, auto

SETTINGS = {"W": (12, 52, 7), "M": (3, 12, 30)}  # horizon, season, days per step
ROOT = Path(__file__).resolve().parents[3]  # repo root, so the app finds its models whatever folder it is started from
TCN_FILE = ROOT / "data/models/dl_weekly_tcn_serve.pt"
TCN_CONFIG = ROOT / "docs/results/dl_weekly.json"
MIN_WEEKS_FOR_TCN = 64  # 52-week lookback + 12-week backtest
FOUNDER_DATASET = 1     # the TCN's dataset flag: founders' series look like the sparse product series, not stores


def _auto_backtest(hist: np.ndarray, h: int, season: int) -> float:
    """Error (MAE) of the method Auto would pick, on the last h periods."""
    inner, check = hist[:-h], hist[-h:]
    errs = [np.mean(np.abs(check - p)) for p in (fn(inner, h, season) for fn in METHODS.values()) if np.isfinite(p).all()]
    return float(min(errs)) if errs else np.inf


@lru_cache(maxsize=1)
def _tcn():
    return load_tcn(TCN_FILE)


def load_tcn(path: Path):
    if not path.exists() or not TCN_CONFIG.exists():
        return None
    import json

    import torch

    from startupiq.dl import windows as W
    from startupiq.dl.train import Config, build_model
    cfg = Config(**json.loads(TCN_CONFIG.read_text(encoding="utf-8"))["chosen"]["tcn"]["config"])
    dummy = W.Windows(np.zeros((1, W.L, 3), np.float32), np.zeros((1, 12, 4), np.float32), np.zeros((1, 1), np.float32),
                      np.zeros((1, 12), np.float32), np.ones(1), np.array(["x"]), np.zeros(1, int))
    model = build_model(cfg, dummy, W.L)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=next(model.parameters()).device))
    return model.eval()


def _tcn_quantiles(model, y: np.ndarray, weeks: pd.DatetimeIndex, origin: int) -> np.ndarray:
    from startupiq.dl import windows as W
    from startupiq.dl.train import predict
    ext = weeks.append(pd.date_range(weeks[-1] + pd.Timedelta(days=7), periods=12, freq="W-SUN"))
    yy = np.concatenate([y, np.zeros(max(0, origin + 12 - len(y)))])
    p, f, s, t, sc = W.window(yy, ext, origin, FOUNDER_DATASET)
    w = W.Windows(p[None].astype(np.float32), f[None].astype(np.float32), s[None].astype(np.float32), np.nan_to_num(t)[None].astype(np.float32),
                  np.array([sc]), np.array(["founder"]), np.array([origin]))
    return predict(model, w)[0]  # [12, 3]


def _wape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(actual - pred).sum() / max(np.abs(actual).sum(), 1e-9) * 100)


TCN_LABEL = "Deep model (TCN)"
TYPED_LABEL = "Deep model that knows your business type"


def forecast(series: SalesSeries, type_probs: list[float] | None = None) -> Forecast:
    """Statistics and up to two deep models each forecast the last h periods without seeing them; the closest forecasts the next h.

    type_probs: the business-type classifier's probabilities (category.py); with them the category-aware network enters too."""
    h, season, step = SETTINGS.get(series.frequency, SETTINGS["W"])
    v = np.asarray(series.values, dtype=float)
    p50, p10, p90, method = auto(v, h, season)
    model_name, chosen = f"auto: {method}", method
    inner, check = v[:-h], v[-h:]
    backtest = {name: _wape(check, p) for name, fn in METHODS.items() if np.isfinite(p := fn(inner, h, season)).all()} if len(inner) >= 4 else {}
    deep = []  # (label, forecast function of origin)
    if series.frequency == "W":
        weeks = pd.DatetimeIndex(pd.to_datetime(series.dates))
        model = _tcn() if len(v) >= MIN_WEEKS_FOR_TCN else None
        if model is not None:
            deep.append((TCN_LABEL, lambda o: _tcn_quantiles(model, v, weeks, o)))
        elif len(v) >= MIN_WEEKS_FOR_TCN:
            model_name += " (deep model file not found: run python -m startupiq.dl.serve_train)"
        if type_probs is not None:
            from startupiq.dl import category_forecast as cf
            if cf.load() is not None and len(v) - h >= cf.L:
                deep.append((TYPED_LABEL, lambda o: cf.quantiles(v, weeks, o, type_probs)))
    best_err = _auto_backtest(v, h, season) if deep else np.inf
    for label, fn in deep:
        back = fn(len(v) - h)
        backtest[label] = _wape(check, back[:, 1])
        err = float(np.mean(np.abs(check - back[:, 1])))
        if err < best_err:
            q = fn(len(v))
            best_err, chosen = err, label
            p10, p50, p90, model_name = q[:, 0], q[:, 1], q[:, 2], f"hybrid: {label} (closest on your last {h} weeks)"
    dates = [series.dates[-1] + timedelta(days=step * (i + 1)) for i in range(h)]
    return Forecast(dates=dates, p10=np.minimum(p10, p50).tolist(), p50=np.asarray(p50).tolist(), p90=np.maximum(p90, p50).tolist(),
                    model=model_name, source=Source.PREDICTED, backtest={k: round(x, 1) for k, x in backtest.items()}, chosen=chosen)
