"""Training windows for the market forecasting models.

Series used for training (one global model learns from all of them):
  - every article's weekly relative series (about 170): the article's share of attention divided by the median of
    all sectors that day, so it sits on the same scale as the sector index
  - the 19 sector series themselves
Evaluation is always on the 19 sector series only.

One window = 52 past weeks -> the next 12 weeks. Each window is scaled by its own level (mean of the 52 past
weeks) and log-transformed, so a big and a small series look alike to the model; forecasts are scaled back.

Inputs per past week:   log(value / level), chatbot interest change, week-of-year (sin, cos), Diwali week, Holi week
Inputs per future week: week-of-year (sin, cos), Diwali week, Holi week (all known in advance)
Plus the sector, as an id the model learns an embedding for.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "market"
L, H = 52, 12
CHATBOTS = {"ChatGPT", "Google Gemini", "Claude (AI)"}
# Festival dates (India). Known years ahead, so they are fair inputs for future weeks.
DIWALI = ["2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08", "2027-10-29"]
HOLI = ["2023-03-08", "2024-03-25", "2025-03-14", "2026-03-04", "2027-03-22"]


def _near(weeks: pd.DatetimeIndex, dates: list[str], days: int = 7) -> np.ndarray:
    d = pd.to_datetime(dates).values
    gap = np.abs(weeks.values[:, None] - d[None, :]).min(axis=1)
    return (gap <= np.timedelta64(days + 3, "D")).astype(np.float32)


def calendar(weeks: pd.DatetimeIndex) -> np.ndarray:
    """(weeks, 4): week-of-year sin, cos, Diwali week, Holi week."""
    woy = weeks.isocalendar().week.to_numpy(dtype=np.float32)
    ang = 2 * np.pi * woy / 52.18
    return np.stack([np.sin(ang), np.cos(ang), _near(weeks, DIWALI), _near(weeks, HOLI)], axis=1).astype(np.float32)


def _weekly(daily: pd.DataFrame) -> pd.DataFrame:
    counts = daily.resample("W-SUN").count()
    return daily.resample("W-SUN").mean().where(counts == 7)


@dataclass
class Data:
    weeks: pd.DatetimeIndex
    values: np.ndarray        # (series, weeks), NaN where a series did not exist yet
    sector_id: np.ndarray     # (series,)
    is_sector: np.ndarray     # (series,) True for the 19 evaluation series
    names: list[str]
    sectors: list[str]
    chatbot: np.ndarray       # (weeks,) log chatbot interest
    cal: np.ndarray           # (weeks + H, 4) calendar, including H weeks after the data ends


def load() -> Data:
    arts = pd.read_parquet(PROC / "articles.parquet")
    daily = pd.read_parquet(PROC / "sectors_daily.parquet")
    weekly_sec = pd.read_parquet(PROC / "sectors_weekly.parquet").pivot(index="week_end", columns="sector", values="index")
    weekly_sec = weekly_sec.sort_index().dropna()
    weeks = weekly_sec.index
    sectors = list(weekly_sec.columns)

    denom = daily.pivot(index="date", columns="sector", values="share_index").median(axis=1)  # median of sectors
    art_daily = arts.pivot_table(index="date", columns=["sector", "project", "title"], values="scaled") * 100
    art_rel = art_daily.div(denom, axis=0)
    art_w = _weekly(art_rel).reindex(weeks)

    chat = arts[arts["title"].isin(CHATBOTS) & (arts["project"] == "en.wikipedia")]
    chat_d = chat.pivot_table(index="date", columns="title", values="scaled").mean(axis=1)
    chatbot = np.log(_weekly(chat_d.to_frame()).iloc[:, 0].reindex(weeks).ffill().bfill().to_numpy() + 1e-3)

    names, values, sid, is_sec = [], [], [], []
    for s in sectors:
        names.append(f"sector:{s}")
        values.append(weekly_sec[s].to_numpy())
        sid.append(sectors.index(s))
        is_sec.append(True)
    for col in art_w.columns:
        names.append(f"{col[1]}:{col[2]}")
        values.append(art_w[col].to_numpy())
        sid.append(sectors.index(col[0]))
        is_sec.append(False)

    future = pd.date_range(weeks[-1] + pd.Timedelta(weeks=1), periods=H, freq="W-SUN")
    return Data(weeks=weeks, values=np.array(values, dtype=np.float64), sector_id=np.array(sid), is_sector=np.array(is_sec),
                names=names, sectors=sectors, chatbot=chatbot.astype(np.float64),
                cal=calendar(weeks.append(future)))


def window(d: Data, i: int, t: int) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Inputs for series i with forecast origin at week index t (last seen week). None if history is incomplete."""
    past = d.values[i, t - L + 1:t + 1]
    if t - L + 1 < 0 or np.isnan(past).any() or (past <= 0).any():
        return None
    level = float(past.mean())
    chat = d.chatbot[t - L + 1:t + 1]
    xp = np.column_stack([np.log(past / level), chat - chat.mean(), d.cal[t - L + 1:t + 1]]).astype(np.float32)
    xf = d.cal[t + 1:t + 1 + H].astype(np.float32)
    return xp, xf, level


def training_set(d: Data, last_target: int) -> dict[str, np.ndarray]:
    """All windows whose 12 target weeks end on or before week index `last_target`."""
    xp, xf, sid, y = [], [], [], []
    for i in range(len(d.names)):
        for t in range(L - 1, last_target - H + 1):
            w = window(d, i, t)
            target = d.values[i, t + 1:t + 1 + H]
            if w is None or np.isnan(target).any() or (target <= 0).any():
                continue
            xp.append(w[0])
            xf.append(w[1])
            sid.append(d.sector_id[i])
            y.append(np.log(target / w[2]).astype(np.float32))
    return {"xp": np.stack(xp), "xf": np.stack(xf), "sid": np.array(sid), "y": np.stack(y)}


def origin_set(d: Data, t: int) -> dict[str, np.ndarray]:
    """One window per sector series at origin t (for validation / test forecasts)."""
    xp, xf, sid, lv = [], [], [], []
    for i in np.where(d.is_sector)[0]:
        w = window(d, i, t)
        assert w is not None, d.names[i]
        xp.append(w[0])
        xf.append(w[1])
        sid.append(d.sector_id[i])
        lv.append(w[2])
    return {"xp": np.stack(xp), "xf": np.stack(xf), "sid": np.array(sid), "level": np.array(lv)}
