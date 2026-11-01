"""Train and compare the three neural forecasters against the baselines.

    python -m market.train                (from ml/, after market.index; uses the GPU when available)

Protocol (the test weeks are touched exactly once):
1. Tuning: train on windows whose targets end by the last training week; forecast the 12 validation weeks for the
   19 sectors. Small grid per model (dropout x epochs), 3 random seeds each, seed forecasts averaged.
2. The setting with the lowest validation MASE is kept for each model. An average of the three models
   ("ensemble") is also scored on validation.
3. Final: retrain each chosen setting on train + validation (3 seeds), forecast the 12 locked test weeks once.
Results: ml/models/runs/market_dl.json, weights: ml/models/market/*.pt (not committed).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from market import dataset, evaluate
from market.dataset import H
from market.models import ARCHS, pinball

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "models" / "runs" / "market_dl.json"
WEIGHTS = ROOT / "models" / "market"
SEEDS = (0, 1, 2)
DROPOUTS = (0.1, 0.3)
CHECKPOINTS = (10, 20, 40, 60)
HIDDEN = {"mlp": 256, "cnn": 64, "lstm": 64}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensors(s: dict) -> tuple[torch.Tensor, ...]:
    return (torch.tensor(s["xp"], device=DEVICE), torch.tensor(s["xf"], device=DEVICE),
            torch.tensor(s["sid"], device=DEVICE, dtype=torch.long))


def predict(model, origin: dict) -> np.ndarray:
    """Quantile forecasts in index points, shape (19, H, 3)."""
    model.eval()
    with torch.no_grad():
        q = model(*tensors(origin)).cpu().numpy()
    return origin["level"][:, None, None] * np.exp(q)


def fit(arch: str, dropout: float, seed: int, train: dict, origin: dict, checkpoints, n_sectors: int):
    """Train one model; return {epoch: forecast} at each checkpoint, and the model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ARCHS[arch](n_sectors, hidden=HIDDEN[arch], dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xp, xf, sid = tensors(train)
    y = torch.tensor(train["y"], device=DEVICE)
    n, out = len(y), {}
    for epoch in range(1, max(checkpoints) + 1):
        model.train()
        for b in torch.randperm(n, device=DEVICE).split(512):
            opt.zero_grad()
            loss = pinball(model(xp[b], xf[b], sid[b]), y[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if epoch in checkpoints:
            out[epoch] = predict(model, origin)
    return out, model


def score(fc: np.ndarray, actual: np.ndarray, scales: np.ndarray) -> dict:
    """Average evaluate.scores over the 19 sectors. fc (19, H, 3), actual (19, H)."""
    per = [evaluate.scores(actual[i], {"p10": fc[i, :, 0], "p50": fc[i, :, 1], "p90": fc[i, :, 2]}, scales[i])
           for i in range(len(actual))]
    return {k: float(np.mean([p[k] for p in per])) for k in per[0]}, per


def main() -> None:
    t0 = time.time()
    d = dataset.load()
    n_weeks, sec = len(d.weeks), np.where(d.is_sector)[0]
    train_end, val_end = n_weeks - 2 * H - 1, n_weeks - H - 1      # last week index of train / of validation
    actual_val = d.values[sec, train_end + 1:val_end + 1]
    actual_test = d.values[sec, val_end + 1:]
    scales = np.array([np.mean(np.abs(np.diff(d.values[i, :train_end + 1]))) for i in sec])
    print(f"device: {DEVICE} | series: {len(d.names)} ({len(sec)} sectors) | weeks: {n_weeks}")

    tr = dataset.training_set(d, train_end)
    o_val = dataset.origin_set(d, train_end)
    print(f"tuning windows: {len(tr['y'])}")

    tuning, chosen = {}, {}
    for arch in ARCHS:
        best = None
        for dropout in DROPOUTS:
            runs = [fit(arch, dropout, s, tr, o_val, CHECKPOINTS, len(d.sectors))[0] for s in SEEDS]
            for ep in CHECKPOINTS:
                fc = np.mean([r[ep] for r in runs], axis=0)
                overall, _ = score(fc, actual_val, scales)
                tuning[f"{arch}|dropout={dropout}|epochs={ep}"] = overall
                if best is None or overall["mase"] < best[0]["mase"]:
                    best = (overall, {"dropout": dropout, "epochs": ep}, fc)
        chosen[arch] = {"setting": best[1], "val": best[0], "val_fc": best[2]}
        print(f"{arch:>5}: best val MASE {best[0]['mase']:.3f} with {best[1]}  ({time.time() - t0:.0f}s)")
    ens_val, _ = score(np.mean([c["val_fc"] for c in chosen.values()], axis=0), actual_val, scales)
    print(f"ensemble: val MASE {ens_val['mase']:.3f}")

    # Final: retrain on train + validation, test once.
    tr_full = dataset.training_set(d, val_end)
    o_test = dataset.origin_set(d, val_end)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    test, test_fc, per_sector = {}, {}, {}
    for arch, c in chosen.items():
        fcs = []
        for s in SEEDS:
            out, model = fit(arch, c["setting"]["dropout"], s, tr_full, o_test, (c["setting"]["epochs"],), len(d.sectors))
            fcs.append(out[c["setting"]["epochs"]])
            torch.save(model.state_dict(), WEIGHTS / f"{arch}_seed{s}.pt")
        test_fc[arch] = np.mean(fcs, axis=0)
        test[arch], per_sector[arch] = score(test_fc[arch], actual_test, scales)
    test_fc["ensemble"] = np.mean([test_fc[a] for a in ARCHS], axis=0)
    test["ensemble"], per_sector["ensemble"] = score(test_fc["ensemble"], actual_test, scales)

    base = json.loads(evaluate.OUT.read_text())
    results = {
        "device": str(DEVICE), "seconds": round(time.time() - t0),
        "windows": {"tuning": len(tr["y"]), "final": len(tr_full["y"])},
        "split": base["split"], "sectors": d.sectors,
        "chosen": {a: {"setting": c["setting"], "val": c["val"]} for a, c in chosen.items()},
        "val": {**{a: c["val"] for a, c in chosen.items()}, "ensemble": ens_val,
                **{b: base["val"][b]["overall"] for b in evaluate.METHODS}},
        "test": {**test, **{b: base["test"][b]["overall"] for b in evaluate.METHODS}},
        "test_per_sector": {m: dict(zip(d.sectors, per_sector[m], strict=True)) for m in per_sector},
        "test_forecast": {m: {s: test_fc[m][i].round(2).tolist() for i, s in enumerate(d.sectors)} for m in test_fc},
        "test_actual": {s: actual_test[i].round(2).tolist() for i, s in enumerate(d.sectors)},
        "tuning": tuning,
    }
    RUNS.write_text(json.dumps(results, indent=2))
    print(f"\n{'model':<16}{'val MASE':>9}{'test MASE':>10}{'1-4w':>7}{'5-8w':>7}{'9-12w':>7}{'cover':>7}")
    for m in [*ARCHS, "ensemble", *evaluate.METHODS]:
        v, t = results["val"][m], results["test"][m]
        print(f"{m:<16}{v['mase']:>9.3f}{t['mase']:>10.3f}{t['mase_1_4']:>7.2f}{t['mase_5_8']:>7.2f}{t['mase_9_12']:>7.2f}"
              f"{t['coverage']:>7.0%}")
    print(f"\nSaved {RUNS.relative_to(ROOT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
