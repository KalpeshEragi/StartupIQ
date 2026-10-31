"""Training and evaluation windows for the weekly deep forecasters (Task 7).

One window = one series at one forecast origin o:
    past    [L, 3]  log1p(y / scale) and week-of-year sin, cos for the L weeks before o
    future  [H, 4]  same week last year log1p(y[t-52] / scale), week-of-year sin, cos, h / H
    static  [1]     dataset (0 Rossmann store, 1 UCI product)
    target  [H]     log1p(y[o:o+H] / scale)
scale = mean of the 12 weeks before o, plus 1: the same scaling as the Step 5 LightGBM,
so one model can serve a drugstore and a gift product. Everything uses y[:o] only,
except the target (leakage checklist W1-W4 applies unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from startupiq.features.splits import WEEK_HORIZON as H

L = 52
SEASON = 52


@dataclass
class Windows:
    past: np.ndarray
    future: np.ndarray
    static: np.ndarray
    target: np.ndarray
    scale: np.ndarray
    series: np.ndarray  # series id per window
    origin: np.ndarray
    mask: np.ndarray | None = None  # [N, H] true where the target counts in the loss (daily: open days with sales)
    ids: np.ndarray | None = None   # [N] integer series index for an embedding (daily: store)

    def __len__(self) -> int:
        return len(self.past)

    def subset(self, keep: np.ndarray) -> "Windows":
        fields = ("past", "future", "static", "target", "scale", "series", "origin", "mask", "ids")
        return Windows(*(None if getattr(self, f) is None else getattr(self, f)[keep] for f in fields))


def _woy(weeks: pd.DatetimeIndex) -> np.ndarray:
    w = weeks.isocalendar().week.to_numpy(dtype=float)
    return np.stack([np.sin(2 * np.pi * w / 52.18), np.cos(2 * np.pi * w / 52.18)], axis=1)


def window(y: np.ndarray, weeks: pd.DatetimeIndex, o: int, ds: int, woy: np.ndarray | None = None) -> tuple:
    woy = _woy(weeks) if woy is None else woy
    scale = float(np.mean(y[o - 12:o])) + 1.0
    past = np.column_stack([np.log1p(np.clip(y[o - L:o], 0, None) / scale), woy[o - L:o]])
    t = np.arange(o, o + H)
    seasonal = np.log1p(np.clip(y[t - SEASON], 0, None) / scale)  # t - 52 < o because H < 52
    fw = woy[np.minimum(t, len(woy) - 1)]
    future = np.column_stack([seasonal, fw, np.arange(1, H + 1) / H])
    target = np.log1p(np.clip(y[o:o + H], 0, None) / scale) if o + H <= len(y) else np.full(H, np.nan)
    return past, future, np.array([ds], float), target, scale


def build(series: dict[str, np.ndarray], weeks: dict[str, pd.DatetimeIndex], origins: dict[str, list[int]]) -> Windows:
    parts = {k: [] for k in ("past", "future", "static", "target", "scale", "series", "origin")}
    for sid, y in series.items():
        woy = _woy(weeks[sid])
        ds = 0 if sid.startswith("ross_") else 1
        for o in origins[sid]:
            p, f, s, t, sc = window(y, weeks[sid], o, ds, woy)
            for k, v in zip(("past", "future", "static", "target", "scale", "series", "origin"), (p, f, s, t, sc, sid, o)):
                parts[k].append(v)
    return Windows(np.asarray(parts["past"], np.float32), np.asarray(parts["future"], np.float32),
                   np.asarray(parts["static"], np.float32), np.asarray(parts["target"], np.float32),
                   np.asarray(parts["scale"], np.float64), np.asarray(parts["series"]), np.asarray(parts["origin"]))


def training_origins(series: dict[str, np.ndarray], eval_origin: dict[str, int], step: int = 1) -> dict[str, list[int]]:
    """Every origin whose targets end before the evaluation origin."""
    return {sid: list(range(L, eval_origin[sid] - H + 1, step)) for sid in series}
