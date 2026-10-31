"""Market signals: FRED retail series and Wikipedia page views into one monthly table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from startupiq.clean.log import CleaningLog

FRED = Path("data/raw/fred")
WIKI = Path("data/raw/wiki")


def run(out_dir: Path = Path("data/processed")) -> CleaningLog:
    log = CleaningLog("signals")
    rows = []
    for f in sorted(FRED.glob("*.csv")):
        s = pd.read_csv(f, parse_dates=["observation_date"])
        s = s.rename(columns={"observation_date": "month", f.stem: "value"})
        s["value"] = pd.to_numeric(s.value, errors="coerce")
        before = len(s)
        s = s.dropna()
        log.record(f"fred_{f.stem}", before, len(s), "Non-numeric values ('.') dropped")
        rows.append(s.assign(source="fred", series=f.stem))
    for f in sorted(WIKI.glob("*.json")):
        items = json.loads(f.read_text(encoding="utf-8")).get("items", [])
        s = pd.DataFrame({"month": pd.to_datetime([i["timestamp"][:8] for i in items]), "value": [i["views"] for i in items]})
        # The API returns the month containing the end date even when it has barely started: drop a trailing month far below the previous one.
        if len(s) > 2 and s.value.iloc[-1] < 0.2 * s.value.iloc[-2]:
            s = s.iloc[:-1]
        log.record(f"wiki_{f.stem}", len(items), len(s), "Trailing partial month dropped if present")
        rows.append(s.assign(source="wikipedia", series=f.stem))
    out = pd.concat(rows, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "signals_monthly.parquet", index=False)
    log.save(out_dir)
    return log
