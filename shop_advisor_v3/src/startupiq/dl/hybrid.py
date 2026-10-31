"""Hybrid candidate: Auto, with the TCN added as one more method it can pick (decision 0015).

Why: in the weekly benchmark the TCN beats every Step 5 method on sparse products (UCI) but
loses badly to seasonal methods on stores. Auto already picks a method per series by its own
last 12 weeks; the hybrid lets it pick the TCN too.

No leakage: the backtest needs a TCN trained only on data before the backtest window, so a
second TCN is trained with its evaluation origin 12 weeks earlier. The forecast itself uses a
TCN trained up to the real origin.

python -m startupiq.models.run dl_hybrid           validation origin
python -m startupiq.models.run dl_hybrid --final   test origin, once
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from startupiq.dl import windows as W
from startupiq.dl.train import Config, fit, predict
from startupiq.evaluate import report
from startupiq.evaluate.metrics import mase, mase_scale, mean_sd
from startupiq.features import weekly as fw
from startupiq.features.splits import SEEDS, WEEK_HORIZON as H
from startupiq.models.forecast_methods import METHODS, auto


def _tcn_forecasts(series, weeks, origin: dict[str, int], cfg: Config, seed: int = 0) -> dict[str, np.ndarray]:
    train = W.build(series, weeks, W.training_origins(series, origin))
    ev = W.build(series, weeks, {s: [o] for s, o in origin.items()})
    q = predict(fit(train, cfg, seed).model, ev)
    return {sid: q[i] for i, sid in enumerate(ev.series)}


def _auto_inner(hist: np.ndarray) -> tuple[str, float]:
    """The method Auto picks on the last 12 weeks of `hist`, and its error there (MAE)."""
    inner, check = hist[:-H], hist[-H:]
    best, err = "Moving average (4 weeks)", np.inf
    for name, fn in METHODS.items():
        p = fn(inner)
        if np.isfinite(p).all() and np.mean(np.abs(check - p)) < err:
            best, err = name, float(np.mean(np.abs(check - p)))
    return best, err


def run(final: bool = False) -> dict:
    t0 = time.time()
    series, weeks = fw.panel()
    test_o = {s: len(y) - H for s, y in series.items()}
    origin = test_o if final else {s: o - H for s, o in test_o.items()}
    inner_o = {s: o - H for s, o in origin.items()}
    cfg = Config(**json.loads(Path("docs/results/dl_weekly.json").read_text(encoding="utf-8"))["chosen"]["tcn"]["config"])
    per_seed = []
    for seed in SEEDS:
        tcn_inner = _tcn_forecasts(series, weeks, inner_o, cfg, seed)  # trained with targets before the backtest window
        tcn_now = _tcn_forecasts(series, weeks, origin, cfg, seed)     # trained with targets before the real origin
        rows = []
        for sid, y in series.items():
            o = origin[sid]
            hist, actual = y[:o], y[o:o + H]
            scale = mase_scale(hist, fw.SEASON)
            a50, a10, a90, a_method = auto(hist)
            _, a_err = _auto_inner(hist)
            t_err = float(np.mean(np.abs(hist[-H:] - tcn_inner[sid][:, 1])))
            use_tcn = t_err < a_err
            p = tcn_now[sid] if use_tcn else np.stack([a10, a50, a90], axis=1)
            rows.append({"series": sid, "dataset": "Rossmann stores" if sid.startswith("ross_") else "UCI products",
                         "Auto": mase(actual, a50, scale), "TCN": mase(actual, tcn_now[sid][:, 1], scale), "Hybrid": mase(actual, p[:, 1], scale),
                         "picked": "TCN" if use_tcn else a_method, "covered": float(np.mean((actual >= p[:, 0]) & (actual <= p[:, 2])))})
        d = pd.DataFrame(rows)
        per_seed.append(d[np.isfinite(d[["Auto", "TCN", "Hybrid"]]).all(axis=1)])
    summary = []
    for ds in ("Rossmann stores", "UCI products", "Pooled"):
        gs = [d if ds == "Pooled" else d[d.dataset == ds] for d in per_seed]
        summary.append({"dataset": ds, "series": len(gs[0]), "Auto": round(gs[0].Auto.mean(), 3),
                        "TCN alone": mean_sd([g.TCN.mean() for g in gs]), "Hybrid": mean_sd([g.Hybrid.mean() for g in gs]),
                        "TCN picked": f"{np.mean([(g.picked == 'TCN').mean() for g in gs]):.0%}",
                        "Hybrid coverage": mean_sd([g.covered.mean() for g in gs]),
                        "_hybrid_mean": float(np.mean([g.Hybrid.mean() for g in gs])), "_hybrid_sd": float(np.std([g.Hybrid.mean() for g in gs])),
                        "_coverage": float(np.mean([g.covered.mean() for g in gs]))})
    pooled = summary[-1]
    beats = pooled["_hybrid_mean"] < pooled["Auto"] - max(pooled["_hybrid_sd"], 1e-9)
    calibrated = 0.70 <= pooled["_coverage"] <= 0.90
    verdict = (f"Hybrid pooled MASE {pooled['_hybrid_mean']:.3f} ± {pooled['_hybrid_sd']:.3f} vs Auto {pooled['Auto']:.3f}; coverage {pooled['_coverage']:.3f}. "
               + ("Beats Auto by more than one sd with coverage inside 0.70-0.90: qualifies to serve (decision 0005 rule)." if beats and calibrated
                  else "Does not meet the decision 0005 rule."))
    summary = [{k: v for k, v in r.items() if not k.startswith("_")} for r in summary]
    out = {"summary": summary, "verdict": verdict, "seconds": round(time.time() - t0), "split": "test" if final else "validation"}
    name = "dl_hybrid_test" if final else "dl_hybrid"
    report.write(name, f"Hybrid: Auto with the TCN as a candidate ({out['split']})", [
        ("How it works", "Per series, Auto backtests its five methods on the last 12 weeks before the origin; the hybrid adds a TCN trained only on "
                         "data before that window, and uses whichever had the lower error. Forecasts then come from the picked method "
                         "(the TCN retrained up to the origin)."),
        ("MASE (lower is better; 3 seeds)", report.table(pd.DataFrame(summary))),
        ("Verdict", verdict),
    ], out, command=f"python -m startupiq.models.run dl_hybrid{' --final' if final else ''}")
    return out
