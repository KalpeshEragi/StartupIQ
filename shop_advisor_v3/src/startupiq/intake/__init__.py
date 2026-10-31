"""M0 intake: turn any reasonable sales upload into regular series and a data tier.

    read_file("sales.xlsx")            -> IntakeResult
    read_table(df, column_map=None)    -> IntakeResult   (already-loaded table; overrides from the UI)

Steps: read (encodings, delimiters, title rows, sheets) -> detect layout (long or wide)
-> assign column roles by name and content -> build series with checks -> tier.
Design and thresholds: devdocs/decisions/0011-intake-design.md and 0004-data-tiers.md.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from startupiq.contracts import IntakeResult, Tier
from startupiq.intake import layout, signals
from startupiq.intake.columns import CONFIRM_BELOW, assign_roles
from startupiq.intake.reader import RawTable, UnreadableFile, read_bytes, read_path
from startupiq.intake.series import tier_for  # noqa: F401  (re-exported for callers and tests)
from startupiq.intake.series import build

TEMPLATE = Path(__file__).resolve().parents[3] / "configs" / "intake_template.csv"
TIER_RANK = {Tier.REFUSED: 0, Tier.SUMMARY_ONLY: 1, Tier.BASIC: 2, Tier.STANDARD: 3, Tier.FULL: 4}


def read_table(df: pd.DataFrame, column_map: dict[str, str] | None = None, *, today: date | None = None,
               sheet: str | None = None, header_row: int | None = None, notes: list[str] | None = None) -> IntakeResult:
    notes = list(notes or [])
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    shape = "long"
    layout_question = None
    roles = assign_roles(df, column_map)

    if "date" not in roles.column_map:
        wide = layout.periods_across(df)
        if wide is not None:
            df, shape = wide, "wide"
            notes.append("Sheet has one column per period; reshaped to one row per period and product.")
            roles = assign_roles(df, {"date": "Period", "revenue": "Amount", **({"product": "Product"} if "Product" in df else {})})
            roles.confidence = {r: 0.8 for r in roles.column_map}  # we reshaped it, so the mapping is ours, not a guess
    elif all(roles.confidence.get(r, 0) < CONFIRM_BELOW for r in ("revenue", "units")):  # no confidently named amount
        wide = layout.products_across(df, roles.column_map["date"])
        if wide is not None:
            df, shape = wide, "wide"
            products = list(df.Product.unique())
            notes.append("Sheet has one column per product; reshaped to one row per date and product.")
            roles = assign_roles(df, {"date": "Date", "revenue": "Amount", "product": "Product"})
            layout_question = (f"We read the columns {', '.join(map(str, products))} as the sales of each product. "
                               "If they are something else (for example prices or stock), tell us.")

    built = build(df, roles.column_map, today=today)
    try:
        sig = signals.extract(df, roles.column_map) if built.tier != Tier.REFUSED and shape == "long" else None
    except (KeyError, ValueError, TypeError):  # the classifier is optional; never fail an upload over it
        sig = None
    confirm = set(roles.needs_confirmation) | set(built.confirm)
    messages = list(built.messages)
    if layout_question:
        confirm.add("layout")
        messages.insert(0, layout_question)
    confirm = sorted(confirm)
    low = [r for r in roles.needs_confirmation if r in roles.column_map]
    if low:
        guesses = ", ".join(f"{r} = '{roles.column_map[r]}'" for r in low)
        messages.insert(0, f"Please check our guess: {guesses}.")
    if built.tier == Tier.REFUSED:
        messages.append("Tip: download the template (date, product, amount) and paste your sales into it.")
    return IntakeResult(
        tier=built.tier, series=built.series, column_map=roles.column_map, confidence=roles.confidence,
        needs_confirmation=confirm, layout=shape, granularity=built.granularity, sheet=sheet, header_row=header_row,
        cleaning_log=notes + built.log, messages=messages, signals=sig,
    )


def _choose(tables: list[RawTable], column_map: dict[str, str] | None, today: date | None) -> IntakeResult:
    results = []
    for t in tables:
        if t.frame.empty:
            continue
        r = read_table(t.frame, column_map, today=today, sheet=t.sheet, header_row=t.header_row, notes=t.notes)
        results.append((TIER_RANK[r.tier], sum(len(s.values) for s in r.series[:1]), r))
    if not results:
        return IntakeResult(tier=Tier.REFUSED, messages=["The file has no data."])
    results.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = results[0][2]
    if len(tables) > 1 and best.sheet is not None:
        best.cleaning_log.insert(0, f"Used sheet '{best.sheet}' (best of {len(tables)} sheets).")
    return best


def read_file(path: str | Path, column_map: dict[str, str] | None = None, *, today: date | None = None) -> IntakeResult:
    try:
        return _choose(read_path(path), column_map, today)
    except UnreadableFile as exc:
        return IntakeResult(tier=Tier.REFUSED, messages=[str(exc)])


def read_upload(data: bytes, filename: str, column_map: dict[str, str] | None = None, *, today: date | None = None) -> IntakeResult:
    try:
        return _choose(read_bytes(data, filename), column_map, today)
    except UnreadableFile as exc:
        return IntakeResult(tier=Tier.REFUSED, messages=[str(exc)])


read_sales = read_table  # name used by the pipeline since Step 0
