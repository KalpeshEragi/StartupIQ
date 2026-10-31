"""Task 7 experiment: deep forecasters on Rossmann daily vs the Step 5 random forest (decision 0005).

python -m startupiq.models.run dl_rossmann           validation origin 2015-05-09
python -m startupiq.models.run dl_rossmann --final   test origin 2015-06-20, once

Windows: 84-day lookback, 42-day horizon, one origin every 10 days (so origins fall on every weekday).
    past    [84, 7]   log1p(sales / scale) (0 when closed), open, promo, state holiday, school holiday, weekday sin/cos
    future  [42, 11]  open, promo, state holiday, school holiday, weekday sin/cos, day-of-year sin/cos, h/42,
                      sales 364 days earlier (log1p / scale) and a flag when that is missing
    static  [9]       store type and assortment (one-hot), log competition distance, Promo2; plus a store embedding
The opening, promotion and holiday calendars are planned in advance (leakage checklist R5); Customers is never used.
Loss counts only open days with sales (mask), as RMSPE does.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from startupiq.dl.models import n_params
from startupiq.dl.train import Config, fit, predict
from startupiq.dl.windows import Windows
from startupiq.evaluate import report
from startupiq.evaluate.metrics import coverage, mean_sd, rmspe
from startupiq.features.splits import ROSS_HORIZON_DAYS as H, ROSS_TEST_START, ROSS_VAL_START, SEEDS

L = 84
STEP = 10  # must not be a multiple of 7: with 14, every training origin was a Tuesday, the model learned the weekday
           # from position alone and was 30-60% off at the Saturday validation origin (decision 0015)
MODELS = Path("data/models")
LOG = Path("docs/results/dl_rossmann_progress.log")
CONFIGS = {"mlp": Config("mlp", hidden=128, dropout=0.1, max_epochs=25, patience=4),
           "lstm": Config("lstm", hidden=64, layers=1, dropout=0.1, attention=True, max_epochs=25, patience=4),
           "tcn": Config("tcn", hidden=32, layers=6, dropout=0.1, max_epochs=25, patience=4)}


def _log(msg: str) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def arrays() -> dict:
    d = pd.read_parquet("data/processed/rossmann_daily.parquet")
    dates = pd.date_range(d.Date.min(), d.Date.max())
    stores = np.sort(d.Store.unique())
    idx = pd.MultiIndex.from_product([stores, dates], names=["Store", "Date"])
    g = d.set_index(["Store", "Date"]).reindex(idx)
    shape = (len(stores), len(dates))
    out = {"stores": stores, "dates": dates,
           "sales": g.Sales.to_numpy(float).reshape(shape),             # NaN where the source has no row
           "present": g.Open.notna().to_numpy().reshape(shape),
           "open": g.Open.fillna(0).to_numpy(float).reshape(shape),
           "promo": g.Promo.fillna(0).to_numpy(float).reshape(shape),
           "state": (g.StateHoliday.fillna("0") != "0").to_numpy(float).reshape(shape),
           "school": g.SchoolHoliday.fillna(0).to_numpy(float).reshape(shape)}
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()
    out["cal"] = np.column_stack([np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                                  np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)])
    st = d.drop_duplicates("Store").set_index("Store").loc[stores]
    dist = np.log1p(st.CompetitionDistance.fillna(st.CompetitionDistance.median())) / 10
    out["static"] = np.column_stack([pd.get_dummies(st.StoreType).reindex(columns=list("abcd"), fill_value=0).to_numpy(float),
                                     pd.get_dummies(st.Assortment).reindex(columns=list("abc"), fill_value=0).to_numpy(float),
                                     dist.to_numpy(), st.Promo2.to_numpy(float)]).astype(np.float32)
    return out


def windows(a: dict, origins: list[int], require_complete: bool) -> Windows:
    parts = {k: [] for k in ("past", "future", "static", "target", "scale", "series", "origin", "mask", "ids")}
    sales, present = a["sales"], a["present"]
    for s in range(len(a["stores"])):
        for o in origins:
            lo, hi = o - L, o + H
            if require_complete and not present[s, lo:hi].all():
                continue  # training windows that touch the 2014 gap are skipped
            past_sales = sales[s, lo:o]
            open_sales = past_sales[(a["open"][s, lo:o] == 1) & (past_sales > 0)]
            if len(open_sales) < 20:
                continue
            scale = float(open_sales.mean()) + 1.0
            ps = np.where((a["open"][s, lo:o] == 1) & (past_sales > 0), np.log1p(np.nan_to_num(past_sales) / scale), 0.0)
            past = np.column_stack([ps, a["open"][s, lo:o], a["promo"][s, lo:o], a["state"][s, lo:o], a["school"][s, lo:o], a["cal"][lo:o, :2]])
            t = np.arange(o, hi)
            ly = sales[s, t - 364]
            ly_ok = np.isfinite(ly) & (ly > 0)
            future = np.column_stack([a["open"][s, t], a["promo"][s, t], a["state"][s, t], a["school"][s, t], a["cal"][t],
                                      np.arange(1, H + 1) / H, np.where(ly_ok, np.log1p(np.nan_to_num(ly) / scale), 0.0), (~ly_ok).astype(float)])
            target_sales = sales[s, t]
            mask = (a["open"][s, t] == 1) & np.isfinite(target_sales) & (target_sales > 0)
            target = np.where(mask, np.log1p(np.nan_to_num(target_sales) / scale), 0.0)
            for k, v in zip(parts, (past, future, a["static"][s], target, scale, str(a["stores"][s]), o, mask, s)):
                parts[k].append(v)
    return Windows(np.asarray(parts["past"], np.float32), np.asarray(parts["future"], np.float32), np.asarray(parts["static"], np.float32),
                   np.asarray(parts["target"], np.float32), np.asarray(parts["scale"]), np.asarray(parts["series"]),
                   np.asarray(parts["origin"]), np.asarray(parts["mask"]), np.asarray(parts["ids"]))


def _evaluate(model, a: dict, ev: Windows) -> dict:
    q = predict(model, ev)                               # [stores, 42, 3] in sales units
    actual = np.stack([a["sales"][i, o:o + H] for i, o in zip(ev.ids, ev.origin)])
    m = ev.mask
    return {"rmspe": rmspe(actual[m], q[..., 1][m]), "coverage": coverage(actual[m], q[..., 0][m], q[..., 2][m]), "rows": int(m.sum())}


def _setup(origin_date: pd.Timestamp):
    a = arrays()
    o = int(np.where(a["dates"] == origin_date)[0][0])
    train = windows(a, list(range(L, o - H + 1, STEP)), require_complete=True)
    ev = windows(a, [o], require_complete=False)
    return a, train, ev


def run(final: bool = False) -> dict:
    t0 = time.time()
    LOG.write_text("", encoding="utf-8") if not final else None
    origin = ROSS_TEST_START if final else ROSS_VAL_START
    a, train, ev = _setup(origin)
    _log(f"{'test' if final else 'validation'}: {len(train):,} training windows, {len(ev)} stores scored")
    rows = []
    for arch, cfg in CONFIGS.items():
        runs = []
        for seed in SEEDS:
            t = fit(train, cfg, seed, lookback=L, n_ids=len(a["stores"]))
            r = _evaluate(t.model, a, ev)
            r.update(epochs=t.epochs, seconds=round(t.seconds), params=n_params(t.model))
            runs.append(r)
            _log(f"{cfg.label()} seed {seed}: RMSPE {r['rmspe']:.4f}, coverage {r['coverage']:.3f}, {t.epochs} epochs, {t.seconds:.0f}s")
            if seed == SEEDS[0]:
                MODELS.mkdir(parents=True, exist_ok=True)
                torch.save(t.model.state_dict(), MODELS / f"dl_rossmann_{arch}{'_test' if final else ''}.pt")
        rows.append({"model": cfg.label(), "RMSPE": mean_sd([r["rmspe"] for r in runs]), "P10-P90 coverage": mean_sd([r["coverage"] for r in runs]),
                     "store-days scored": runs[0]["rows"], "epochs": runs[0]["epochs"], "seconds per run": runs[0]["seconds"], "params": runs[0]["params"],
                     "_mean": float(np.mean([r["rmspe"] for r in runs])), "_sd": float(np.std([r["rmspe"] for r in runs]))})
    bar_file = Path("docs/results/forecast_rossmann_test.json" if final else "docs/results/forecast_rossmann.json")
    bar_rows = json.loads(bar_file.read_text(encoding="utf-8"))["table"]
    bar = {r["model"]: r["rmspe"] for r in bar_rows if r.get("split", "test" if final else "validation") == ("test" if final else "validation")}
    rf = next(v for k, v in bar.items() if k.startswith("Random forest"))
    best = min(rows, key=lambda r: r["_mean"])
    verdict = (f"Best deep model {best['model']}: RMSPE {best['_mean']:.4f} ± {best['_sd']:.4f} vs random forest {rf:.4f}. "
               + ("Beats the forest by more than one sd." if best["_mean"] < rf - max(best["_sd"], 1e-9) else "Does not beat the forest by more than one sd."))
    table = pd.DataFrame([{"model": k, "RMSPE": f"{v:.4f}", "source": "Step 5"} for k, v in bar.items()]
                         + [{"model": r["model"], "RMSPE": r["RMSPE"], "source": "this run, 3 seeds"} for r in rows])
    out = {"rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows], "step5": bar, "verdict": verdict,
           "configs": {k: asdict(v) for k, v in CONFIGS.items()}, "seconds": round(time.time() - t0)}
    name = "dl_rossmann_test" if final else "dl_rossmann"
    report.write(name, f"Deep forecasters on Rossmann daily ({'test window, scored once' if final else 'validation'})", [
        ("Setup", f"{len(train):,} training windows (84-day lookback, 42-day horizon, origins every {STEP} days, windows touching the 2014 gap skipped); "
                  f"{len(ev)} stores scored at {origin.date()}. Settings fixed in advance (moderate size, dropout 0.1, weight decay 1e-4); no separate search on this split."),
        ("Against Step 5 (RMSPE on open days with sales; lower is better)", report.table(table)),
        ("Verdict", verdict),
        ("Deep models, 3 seeds", report.table(pd.DataFrame(out["rows"]))),
    ], out, command=f"python -m startupiq.models.run dl_rossmann{' --final' if final else ''}")
    return out
