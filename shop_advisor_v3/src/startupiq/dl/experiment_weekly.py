"""Task 7 experiment: deep forecasters on the weekly panel vs the Step 5 bar (decision 0005).

python -m startupiq.models.run dl_weekly           validation: search, seeds, ablations
python -m startupiq.models.run dl_weekly --final   test origin, once, with the recorded choices

Same series, origins, horizon, MASE scale and baselines as models/forecast_weekly.py.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from startupiq.dl import windows as W
from startupiq.dl.models import n_params
from startupiq.dl.train import Config, Trained, fit, predict
from startupiq.evaluate import report
from startupiq.evaluate.metrics import coverage, mase, mase_scale, mean_sd
from startupiq.features import weekly as fw
from startupiq.features.splits import SEEDS, WEEK_HORIZON as H

MODELS = Path("data/models")
LOG = Path("docs/results/dl_weekly_progress.log")
# Trained on the GPU (RTX 2050, CUDA build of PyTorch). A first attempt on CPU took 2.5 hours for one LSTM and was
# stopped; on the GPU the same model takes about 6 minutes, so the full search below fits in about an hour.
SEARCH_PER_ARCH = 5
ORIGIN_STEP = 1
MAX_EPOCHS = 30
GRID = {"hidden": [32, 64, 128], "layers": [1, 2], "dropout": [0.0, 0.1, 0.3], "lr": [1e-3, 3e-3], "weight_decay": [0.0, 1e-4, 1e-3]}
STEP5 = Path("docs/results/forecast_weekly.json")
STEP5_TEST = Path("docs/results/forecast_weekly_test.json")
CONFIDENCE_BAND = (0.70, 0.90)


def _log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def candidates(arch: str, n: int, seed: int = 0) -> list[Config]:
    """Random search: n distinct configurations drawn from GRID (fixed seed, so reproducible)."""
    rng = np.random.default_rng(seed + {"mlp": 1, "lstm": 2, "tcn": 3}[arch])
    out, seen = [], set()
    while len(out) < n:
        c = {k: v[rng.integers(len(v))] for k, v in GRID.items()}
        if arch == "tcn":
            c["layers"], c["hidden"] = 6, min(c["hidden"], 64)  # 6 dilated blocks give a 127-week receptive field
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            out.append(Config(arch, max_epochs=MAX_EPOCHS, patience=4,
                              **{k: (float(v) if k in ("dropout", "lr", "weight_decay") else int(v)) for k, v in c.items()}))
    return out


def score(series, origin: dict[str, int], q: np.ndarray, sids: np.ndarray) -> dict:
    rows = []
    for i, sid in enumerate(sids):
        y, o = series[sid], origin[sid]
        actual = y[o:o + H]
        rows.append({"series": sid, "dataset": "Rossmann stores" if sid.startswith("ross_") else "UCI products",
                     "mase": mase(actual, q[i, :, 1], mase_scale(y[:o], fw.SEASON)),
                     "covered": float(np.mean((actual >= q[i, :, 0]) & (actual <= q[i, :, 2])))})
    d = pd.DataFrame(rows)
    d = d[np.isfinite(d.mase)]
    out = {"pooled": float(d.mase.mean()), "coverage": float(d.covered.mean())}
    for ds, g in d.groupby("dataset"):
        out[ds] = float(g.mase.mean())
        out[ds + " coverage"] = float(g.covered.mean())
    return out


def _run(cfg: Config, seed: int, train: W.Windows, ev: W.Windows, series, origin) -> tuple[dict, Trained]:
    t = fit(train, cfg, seed)
    s = score(series, origin, predict(t.model, ev), ev.series)
    s.update(config=cfg.label(), seed=seed, epochs=t.epochs, seconds=round(t.seconds), params=n_params(t.model))
    _log(f"{cfg.label()} seed {seed}: pooled MASE {s['pooled']:.3f}, coverage {s['coverage']:.3f}, {t.epochs} epochs, {t.seconds:.0f}s")
    return s, t


def _setup(final: bool):
    series, weeks = fw.panel()
    test_o = {s: len(y) - H for s, y in series.items()}
    origin = test_o if final else {s: o - H for s, o in test_o.items()}
    train = W.build(series, weeks, W.training_origins(series, origin, step=ORIGIN_STEP))
    ev = W.build(series, weeks, {s: [o] for s, o in origin.items()})
    return series, origin, train, ev


def _bar(path: Path) -> dict:
    if not path.exists():
        return {}
    d = json.loads(path.read_text(encoding="utf-8"))
    summary = (d.get("test") or d.get("validation"))["summary"]
    pooled = next(r for r in summary if r["dataset"] == "Pooled")
    return {"Auto": float(pooled["Auto: best method on the series' own last 12 weeks"]),
            "LightGBM global": float(str(pooled["LightGBM global"]).split(" ")[0]),
            "Moving average (4 weeks)": float(pooled["Moving average (4 weeks)"])}


def run(final: bool = False) -> dict:
    return run_test() if final else run_validation()


def run_validation() -> dict:
    t0 = time.time()
    LOG.write_text("", encoding="utf-8")
    series, origin, train, ev = _setup(final=False)
    _log(f"{len(train)} training windows, {len(ev)} evaluation series")
    rows, best = [], {}
    for arch in ("mlp", "lstm", "tcn"):
        for cfg in candidates(arch, SEARCH_PER_ARCH):
            s, t = _run(cfg, 0, train, ev, series, origin)
            rows.append({"stage": "search", **s})
            if arch not in best or s["pooled"] < best[arch][0]["pooled"]:
                best[arch] = (s, t)
    seeds_rows, chosen = [], {}
    MODELS.mkdir(parents=True, exist_ok=True)
    for arch, (s0, t0_) in best.items():
        runs = [s0] + [_run(t0_.config, seed, train, ev, series, origin)[0] for seed in SEEDS[1:]]
        seeds_rows.append({"model": t0_.config.label(), "arch": arch,
                           "pooled MASE": mean_sd([r["pooled"] for r in runs]),
                           "Rossmann stores": mean_sd([r["Rossmann stores"] for r in runs]),
                           "UCI products": mean_sd([r["UCI products"] for r in runs]),
                           "P10-P90 coverage": mean_sd([r["coverage"] for r in runs]),
                           "params": s0["params"], "epochs": s0["epochs"]})
        chosen[arch] = {"config": asdict(t0_.config), "pooled_mean": float(np.mean([r["pooled"] for r in runs])),
                        "pooled_sd": float(np.std([r["pooled"] for r in runs])), "coverage_mean": float(np.mean([r["coverage"] for r in runs]))}
        torch.save(t0_.model.state_dict(), MODELS / f"dl_weekly_{arch}.pt")
    # Ablation 1: does attention help the LSTM?
    lstm_cfg = best["lstm"][1].config
    no_att = Config(**{**asdict(lstm_cfg), "attention": not lstm_cfg.attention})
    abl_att = [{"variant": lstm_cfg.label(), "pooled MASE": best["lstm"][0]["pooled"]},
               {"variant": no_att.label(), "pooled MASE": _run(no_att, 0, train, ev, series, origin)[0]["pooled"]}]
    # Ablation 2: regularisation (dropout and weight decay) on the best LSTM settings.
    reg = []
    for do, wd in ((0.0, 0.0), (0.3, 0.0), (0.0, 1e-3), (0.3, 1e-3)):
        c = Config(**{**asdict(lstm_cfg), "dropout": do, "weight_decay": wd})
        s, t = _run(c, 0, train, ev, series, origin)
        gap = t.history[-1][1] - t.history[-1][0] if t.history else np.nan
        reg.append({"dropout": do, "weight decay": wd, "pooled MASE": s["pooled"], "coverage": s["coverage"], "epochs": t.epochs,
                    "holdout minus train loss (last epoch)": float(gap)})

    bar = _bar(STEP5)
    best_arch = min(chosen, key=lambda a: chosen[a]["pooled_mean"])
    b = chosen[best_arch]
    beats = b["pooled_mean"] < bar.get("Auto", np.inf) - max(b["pooled_sd"], 1e-9)
    calibrated = CONFIDENCE_BAND[0] <= b["coverage_mean"] <= CONFIDENCE_BAND[1]
    verdict = ("serves" if beats and calibrated else "stays an experiment") + \
              f" (best deep model {best_arch}: pooled MASE {b['pooled_mean']:.3f} ± {b['pooled_sd']:.3f}, coverage {b['coverage_mean']:.3f}; " \
              f"Auto {bar.get('Auto', float('nan')):.3f}; rule: beat Auto by more than one sd AND coverage within {CONFIDENCE_BAND})"
    out = {"search": rows, "chosen": chosen, "seeds": seeds_rows, "attention_ablation": abl_att, "regularisation": reg,
           "step5_bar": bar, "verdict": verdict, "best_arch": best_arch, "seconds": round(time.time() - t0)}
    comparison = pd.DataFrame([{"model": k, "pooled MASE": f"{v:.3f}", "source": "Step 5 (decision 0012)"} for k, v in bar.items()]
                              + [{"model": r["model"], "pooled MASE": r["pooled MASE"], "source": "this run, 3 seeds"} for r in seeds_rows])
    report.write("dl_weekly", "Deep forecasters on the weekly panel (Task 7)", [
        ("Setup", f"{len(train):,} training windows (52-week lookback, 12-week horizon), {len(ev):,} series scored at the validation origin. "
                  f"Quantile (pinball) loss for P10/P50/P90; AdamW; gradient clipping; early stopping on 10% of series held out of training. "
                  f"Random search: {SEARCH_PER_ARCH} settings per architecture (seed 0), then 3 seeds for the best."),
        ("Against the Step 5 bar (validation, pooled MASE; lower is better)", report.table(comparison)),
        ("Verdict (decision rule in 0005)", verdict),
        ("Best settings per architecture, 3 seeds", report.table(pd.DataFrame(seeds_rows))),
        ("Ablation: attention in the LSTM", report.table(pd.DataFrame(abl_att))),
        ("Ablation: regularisation (best LSTM settings, seed 0)", report.table(pd.DataFrame(reg))),
        ("Random search (seed 0)", report.table(pd.DataFrame(rows)[["config", "pooled", "Rossmann stores", "UCI products", "coverage", "epochs", "seconds", "params"]])),
    ], out, command="python -m startupiq.models.run dl_weekly")
    return out


def run_test() -> dict:
    """Retrain the chosen settings (3 seeds) on everything before the test origin; score the test origin once."""
    t0 = time.time()
    chosen = json.loads(Path("docs/results/dl_weekly.json").read_text(encoding="utf-8"))["chosen"]
    series, origin, train, ev = _setup(final=True)
    rows = []
    for arch, c in chosen.items():
        cfg = Config(**c["config"])
        runs = [_run(cfg, seed, train, ev, series, origin) for seed in SEEDS]
        torch.save(runs[0][1].model.state_dict(), MODELS / f"dl_weekly_{arch}_test.pt")
        rs = [r[0] for r in runs]
        rows.append({"model": cfg.label(), "pooled MASE": mean_sd([r["pooled"] for r in rs]),
                     "Rossmann stores": mean_sd([r["Rossmann stores"] for r in rs]), "UCI products": mean_sd([r["UCI products"] for r in rs]),
                     "P10-P90 coverage": mean_sd([r["coverage"] for r in rs])})
    bar = _bar(STEP5_TEST)
    comparison = pd.DataFrame([{"model": k, "pooled MASE": f"{v:.3f}", "source": "Step 5 test (decision 0012)"} for k, v in bar.items()]
                              + [{"model": r["model"], "pooled MASE": r["pooled MASE"], "source": "test, 3 seeds"} for r in rows])
    out = {"test": rows, "step5_test_bar": bar, "seconds": round(time.time() - t0)}
    report.write("dl_weekly_test", "Deep forecasters: weekly test origin (scored once)", [
        ("Against the Step 5 test bar (pooled MASE)", report.table(comparison)),
        ("Per dataset and coverage", report.table(pd.DataFrame(rows))),
    ], out, command="python -m startupiq.models.run dl_weekly --final")
    return out
