"""Category-aware deep forecasters on the SYNTHETIC businesses: does knowing the business type help the forecast?

python -m startupiq.dl.category_forecast

MLP, LSTM (+attention) and 1D CNN (TCN) from dl/models.py, each trained twice on the training shops:
  aware  static input = the business type as six probabilities (one-hot in training)
  blind  static input = zeros (same network, nothing about the type)
Inputs per window: the last 26 weeks (scaled), week of year, a Diwali-weeks flag, and for the 12 weeks
ahead the same calendar features plus last year's sales when the shop has them. Choice of architecture on
the validation shops (true type); the test shops are scored once, with the type as the classifier predicts
it from the upload (category.py), i.e. end to end. Baselines: moving average, statistical Auto, and the
category's average seasonal shape scaled to the shop.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from startupiq.dl.windows import Windows
from startupiq.synthetic import DIWALI, NAMES

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "data" / "models" / "dl_category.pt"
CONFIG = ROOT / "data" / "models" / "dl_category.json"
L, H = 26, 12
SEEDS = (0, 1, 2)
CONFIGS = {"mlp": dict(arch="mlp", hidden=64, dropout=0.1, lr=1e-3, max_epochs=25, patience=4),
           "lstm": dict(arch="lstm", hidden=32, layers=1, dropout=0.1, lr=1e-3, attention=True, max_epochs=25, patience=4),
           "tcn": dict(arch="tcn", hidden=32, layers=5, dropout=0.1, lr=1e-3, max_epochs=25, patience=4)}
ARCH_NAME = {"mlp": "MLP", "lstm": "LSTM + attention", "tcn": "1D CNN (TCN)"}


def calendar(weeks: pd.DatetimeIndex) -> np.ndarray:
    """[n, 3]: week of year as sin/cos, and 1 when the week touches the fortnight before Diwali."""
    w = weeks.isocalendar().week.to_numpy(float)
    flag = np.zeros(len(weeks))
    for d in DIWALI:
        flag = np.maximum(flag, ((weeks >= d - pd.Timedelta(days=14)) & (weeks - pd.Timedelta(days=6) <= d + pd.Timedelta(days=2))).astype(float))
    return np.column_stack([np.sin(2 * np.pi * w / 52.18), np.cos(2 * np.pi * w / 52.18), flag])


def window(y: np.ndarray, weeks: pd.DatetimeIndex, o: int, static: np.ndarray) -> tuple:
    ext = weeks.append(pd.date_range(weeks[-1] + pd.Timedelta(days=7), periods=H, freq="W-SUN")) if o + H > len(weeks) else weeks
    cal = calendar(ext)
    scale = float(np.mean(y[o - 12:o])) + 1.0
    past = np.column_stack([np.log1p(np.clip(y[o - L:o], 0, None) / scale), cal[o - L:o]])
    t = np.arange(o, o + H)
    has_ly = t - 52 >= 0
    ly = np.where(has_ly, np.log1p(np.clip(y[np.clip(t - 52, 0, None)], 0, None) / scale), 0.0)
    future = np.column_stack([cal[t], ly, has_ly.astype(float), np.arange(1, H + 1) / H])
    target = np.log1p(np.clip(y[o:o + H], 0, None) / scale) if o + H <= len(y) else np.full(H, np.nan)
    return past, future, static, target, scale


def build(series: dict, origins: dict, statics: dict) -> Windows:
    parts = [[] for _ in range(7)]
    for b, olist in origins.items():
        y, weeks = series[b]
        for o in olist:
            for i, v in enumerate((*window(y, weeks, o, statics[b]), b, o)):
                parts[i].append(v)
    return Windows(np.asarray(parts[0], np.float32), np.asarray(parts[1], np.float32), np.asarray(parts[2], np.float32),
                   np.asarray(parts[3], np.float32), np.asarray(parts[4], np.float64), np.asarray(parts[5]), np.asarray(parts[6]))


def weekly_series() -> tuple[pd.DataFrame, dict]:
    from startupiq.synthetic import OUT
    meta = pd.read_parquet(OUT / "businesses.parquet")
    daily = pd.read_parquet(OUT / "daily.parquet", columns=["business", "date", "sales"])
    series = {}
    for b, g in daily.groupby("business", sort=True):
        w = g.set_index("date").sales.resample("W-SUN").sum()
        series[b] = (w.to_numpy(float), w.index)
    return meta, series


def season_profiles(meta: pd.DataFrame, series: dict) -> dict[str, np.ndarray]:
    """Average weekly sales over the shop's own trend line, by ISO week, per category (training shops only)."""
    acc = {c: [[] for _ in range(54)] for c in NAMES}
    for b in meta.business[meta.split == "train"]:
        y, weeks = series[b]
        if len(y) < 26:
            continue
        x = np.arange(len(y))
        ratio = (y + 1) / np.exp(np.polyval(np.polyfit(x, np.log1p(y), 1), x))
        cat = meta.category[b]
        for wk, r in zip(weeks.isocalendar().week, ratio):
            acc[cat][int(wk)].append(r)
    out = {}
    for c in NAMES:
        prof = np.array([np.mean(v) if v else 1.0 for v in acc[c]])
        prof[0] = prof[1]
        out[c] = np.convolve(np.r_[prof[-1:], prof, prof[:1]], np.ones(3) / 3, mode="valid")  # light smoothing
    return out


def profile_forecast(y, weeks, prof) -> np.ndarray:
    future = pd.date_range(weeks[-1] + pd.Timedelta(days=7), periods=H, freq="W-SUN")
    wk = lambda idx: prof[np.asarray(idx.isocalendar().week, int)]  # noqa: E731
    base = np.mean(y[-4:] / wk(weeks[-4:]))
    return base * wk(future)


def _score(actual, pred, hist) -> tuple[float, float]:
    scale = np.mean(np.abs(np.diff(hist))) + 1e-9
    return float(np.mean(np.abs(actual - pred)) / scale), float(np.abs(actual - pred).sum() / max(actual.sum(), 1e-9))


def run() -> dict:
    import torch

    from startupiq import category
    from startupiq.dl.train import Config, fit, predict
    from startupiq.evaluate import report
    from startupiq.models.forecast_methods import auto, moving_average

    t0 = time.time()
    meta, series = weekly_series()
    onehot = {b: np.eye(len(NAMES))[NAMES.index(c)].astype(np.float32) for b, c in zip(meta.business, meta.category)}
    zeros = {b: np.zeros(len(NAMES), np.float32) for b in meta.business}
    n = {b: len(series[b][0]) for b in meta.business}
    train_o = {b: list(range(L, n[b] - H + 1, 2)) for b in meta.business[meta.split == "train"] if n[b] >= L + H}
    ev = {s: [b for b in meta.business[meta.split == s] if n[b] >= L + H] for s in ("val", "test")}
    # the type as the classifier sees it from each test shop's upload (the classifier never trained on test shops)
    data = category.dataset()
    clf = category.load()
    probs = clf["model"].predict_proba(data.set_index("business").loc[ev["test"], category.FEATURES].to_numpy(float))
    predicted = {b: p.astype(np.float32) for b, p in zip(ev["test"], probs)}
    clf_acc = float(np.mean([NAMES[int(np.argmax(predicted[b]))] == meta.category[b] for b in ev["test"]]))
    prof = season_profiles(meta, series)

    def baselines(split):
        rows = []
        for b in ev[split]:
            y, weeks = series[b]
            hist, act = y[:-H], y[-H:]
            for name, p in (("Moving average (4 weeks)", moving_average(hist, H)), ("Statistical Auto", auto(hist, H, 52)[0]),
                            ("Category seasonal shape (true type)", profile_forecast(hist, weeks[:-H], prof[meta.category[b]]))):
                rows.append({"business": b, "model": name, **dict(zip(("MASE", "WAPE"), _score(act, p, hist)))})
        return rows

    def deep_rows(model, split, statics, label):
        w = build(series, {b: [n[b] - H] for b in ev[split]}, statics)
        q = predict(model, w)
        out = []
        for i, b in enumerate(w.series):
            y = series[b][0]
            out.append({"business": b, "model": label, **dict(zip(("MASE", "WAPE"), _score(y[-H:], q[i][:, 1], y[:-H]))),
                        "covered": float(np.mean((y[-H:] >= q[i][:, 0]) & (y[-H:] <= q[i][:, 2])))})
        return out

    val_rows, test_rows, runs, saved = baselines("val"), baselines("test"), [], {}
    for arch, kw in CONFIGS.items():
        for mode, statics in (("aware", onehot), ("blind", zeros)):
            train = build(series, train_o, statics)
            for seed in SEEDS:
                t = fit(train, Config(**kw), seed=seed, lookback=L)
                label = f"{ARCH_NAME[arch]}, {'knows the type' if mode == 'aware' else 'type-blind'}"
                v = deep_rows(t.model, "val", statics, label)
                te = deep_rows(t.model, "test", predicted if mode == "aware" else zeros, label)
                te_true = deep_rows(t.model, "test", onehot, label + " (true type)") if mode == "aware" else []
                for r in v + te + te_true:
                    r["seed"] = seed
                val_rows += v
                test_rows += te + te_true
                runs.append({"model": label, "seed": seed, "epochs": t.epochs, "seconds": round(t.seconds)})
                if seed == 0:
                    saved[(arch, mode)] = {k: v.detach().cpu() for k, v in t.model.state_dict().items()}
                print(label, seed, round(pd.DataFrame(v).MASE.mean(), 4), f"{t.epochs} epochs {t.seconds:.0f}s", flush=True)
                del t
                torch.cuda.empty_cache()

    def summary(rows):
        d = pd.DataFrame(rows)
        per_seed = d.groupby(["model", d.get("seed", pd.Series(0, index=d.index)).fillna(0)]).agg(MASE=("MASE", "mean"), WAPE=("WAPE", "mean"))
        s = per_seed.groupby("model").agg(MASE=("MASE", "mean"), MASE_sd=("MASE", "std"), WAPE=("WAPE", "mean")).fillna(0)
        if "covered" in d:
            s["range coverage"] = d.groupby("model").covered.mean()
        return s.sort_values("MASE").reset_index()

    val, test = summary(val_rows), summary(test_rows)
    aware = val[val.model.str.contains("knows the type") & ~val.model.str.contains("true type")]
    best_label = aware.iloc[0].model
    best_arch = next(a for a, nm in ARCH_NAME.items() if best_label.startswith(nm))
    torch.save(saved[(best_arch, "aware")], MODEL)
    CONFIG.write_text(json.dumps({"config": CONFIGS[best_arch], "L": L, "H": H, "classes": NAMES, "label": best_label}, indent=1), encoding="utf-8")

    d = pd.DataFrame(test_rows)
    d["history"] = d.business.map(lambda b: "38-63 weeks (no last year)" if n[b] < 64 else "64+ weeks (has last year)")
    focus = [best_label, best_label.replace("knows the type", "type-blind"), "Statistical Auto", "Category seasonal shape (true type)"]
    by_hist = d[d.model.isin(focus)].groupby(["history", "model"]).MASE.mean().unstack(0).loc[focus].round(3).reset_index()
    out = {"validation": val.round(4).to_dict("records"), "test": test.round(4).to_dict("records"), "chosen": best_label,
           "classifier_test_accuracy": round(clf_acc, 4), "by_history": by_hist.to_dict("records"), "runs": runs,
           "shops": {"train": len(train_o), "val": len(ev["val"]), "test": len(ev["test"])}, "seconds": round(time.time() - t0)}
    fmt = lambda df: df.assign(MASE=df.MASE.map(lambda v: f"{v:.3f}"), MASE_sd=df.MASE_sd.map(lambda v: f"{v:.3f}" if v else ""))  # noqa: E731
    report.write("dl_category", "Category-aware deep forecasters (SYNTHETIC data)", [
        ("Data", f"SYNTHETIC weekly sales. Training windows from {len(train_o)} shops; scored on the last 12 weeks of {len(ev['val'])} validation and "
                 f"{len(ev['test'])} test shops with 38+ weeks of history. MASE: error divided by the error of 'same as last week' on the shop's "
                 "own history (lower is better). 3 seeds per network (mean, sd)."),
        ("Validation (true type; choose here)", report.table(fmt(val))),
        ("Chosen", f"**{best_label}**: lowest validation MASE among the type-aware networks."),
        ("Test, scored once (type predicted by the classifier from the upload)",
         f"The classifier got the type right for {clf_acc:.1%} of these shops.\n\n" + report.table(fmt(test))),
        ("Test MASE by length of history", report.table(by_hist)),
    ], out, command="python -m startupiq.dl.category_forecast")
    return out


@lru_cache(maxsize=1)
def load():
    if not MODEL.exists() or not CONFIG.exists():
        return None
    import torch

    from startupiq.dl.train import Config, build_model
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    weeks = pd.date_range("2024-01-07", periods=L + H, freq="W-SUN")
    dummy = build({0: (np.ones(L + H), weeks)}, {0: [L]}, {0: np.zeros(len(NAMES), np.float32)})
    model = build_model(Config(**spec["config"]), dummy, L)
    model.load_state_dict(torch.load(MODEL, weights_only=True, map_location=next(model.parameters()).device))
    return model.eval(), spec


def quantiles(y: np.ndarray, weeks: pd.DatetimeIndex, origin: int, probs) -> np.ndarray | None:
    """[H, 3] P10/P50/P90 for the 12 weeks after `origin`, given the type probabilities. None if unavailable."""
    got = load()
    if got is None or origin < L:
        return None
    from startupiq.dl.train import predict
    w = build({0: (np.asarray(y, float), weeks)}, {0: [origin]}, {0: np.asarray(probs, np.float32)})
    return predict(got[0], w)[0]


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=str)[:4000])
