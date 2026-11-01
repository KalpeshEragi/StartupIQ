"""Make the live market forecast the app serves.

    python -m market.forecast             (from ml/, after market.train; uses the GPU when available)

1. Retrains MLP, CNN and LSTM on every week up to the last complete one, with the settings chosen on validation
   (models/runs/market_dl.json), 3 seeds each, and averages them (the ensemble, joint best on validation).
2. Forecasts the next 12 weeks for every sector as P10 / P50 / P90, plus exponential smoothing for comparison.
3. Writes backend/models/market_forecast.json (not committed): per sector the last 52 weeks, the forecast, the
   monthly index (for "you vs your market") and how the network and smoothing did on the locked test weeks.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from market import dataset, evaluate
from market.dataset import H
from market.models import ARCHS
from market.sectors import SECTORS
from market.train import RUNS, SEEDS, fit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "backend" / "models" / "market_forecast.json"


def monthly_index() -> dict[str, dict[str, float]]:
    daily = pd.read_parquet(dataset.PROC / "sectors_daily.parquet")
    wide = daily.pivot(index="date", columns="sector", values="index")
    counts = wide.resample("MS").count()
    m = wide.resample("MS").mean().where(counts >= 25)  # complete months only
    return {s: {k.strftime("%Y-%m"): round(float(v), 2) for k, v in m[s].dropna().items()} for s in m.columns}


def main() -> None:
    runs = json.loads(RUNS.read_text())
    d = dataset.load()
    last = len(d.weeks) - 1
    train = dataset.training_set(d, last)
    origin = dataset.origin_set(d, last)
    fcs = []
    for arch in ARCHS:
        setting = runs["chosen"][arch]["setting"]
        for s in SEEDS:
            out, _ = fit(arch, setting["dropout"], s, train, origin, (setting["epochs"],), len(d.sectors))
            fcs.append(out[setting["epochs"]])
        print(f"{arch}: trained on {len(train['y'])} windows")
    fc = np.mean(fcs, axis=0)                                            # (19, H, 3)
    future = pd.date_range(d.weeks[-1] + pd.Timedelta(weeks=1), periods=H, freq="W-SUN")
    months = monthly_index()

    sectors = {}
    for i, key in enumerate(d.sectors):
        hist = d.values[np.where(d.is_sector)[0][i]]
        ets = evaluate.forecast(evaluate.ets, hist)
        recent = float(hist[-4:].mean())
        p50 = fc[i, :, 1]
        sectors[key] = {
            "label": SECTORS[key]["label"],
            "history": [{"week": w.strftime("%Y-%m-%d"), "index": round(float(v), 2)}
                        for w, v in zip(d.weeks[-52:], hist[-52:], strict=True)],
            "forecast": [{"week": w.strftime("%Y-%m-%d"), "p10": round(float(a), 2), "p50": round(float(b), 2),
                          "p90": round(float(c), 2)} for w, (a, b, c) in zip(future, fc[i], strict=True)],
            "smoothing": [{"week": w.strftime("%Y-%m-%d"), "p10": round(float(a), 2), "p50": round(float(b), 2),
                           "p90": round(float(c), 2)}
                          for w, a, b, c in zip(future, ets["p10"], ets["p50"], ets["p90"], strict=True)],
            "change_4w": round(float(p50[:4].mean() / recent - 1), 4),
            "change_12w": round(float(p50[-4:].mean() / recent - 1), 4),
            "change_12w_low": round(float(fc[i, -4:, 0].mean() / recent - 1), 4),
            "change_12w_high": round(float(fc[i, -4:, 2].mean() / recent - 1), 4),
            "peak_week": future[int(np.argmax(p50))].strftime("%Y-%m-%d"),
            "backtest": {"network_mase": round(runs["test_per_sector"]["ensemble"][key]["mase"], 3),
                         "smoothing_mase": round(json.loads(evaluate.OUT.read_text())["test"]["ets"]["sectors"][key]["mase"], 3)},
            "monthly": months.get(key, {}),
        }
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_through": d.weeks[-1].strftime("%Y-%m-%d"),
        "model": "Ensemble of MLP, 1D CNN and LSTM with attention (3 seeds each), trained on 2023-2026 weekly Wikipedia "
                 "page-view interest per sector, corrected for the shift to chatbots.",
        "test": {m: round(runs["test"][m]["mase"], 3) for m in ("ensemble", "lstm", "ets")},
        "sectors": sectors,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"Wrote {OUT} ({len(sectors)} sectors, data through {payload['data_through']}, forecast to {future[-1]:%Y-%m-%d})")


if __name__ == "__main__":
    main()
