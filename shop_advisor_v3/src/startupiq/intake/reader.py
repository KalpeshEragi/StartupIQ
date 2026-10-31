"""Read an uploaded file into raw tables: encodings, delimiters, title rows, sheets, total rows."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from startupiq.intake.parsing import date_share, looks_like_period, to_number

MONTH_HEADER = re.compile(r"^[A-Za-z]{3,9}[\s\-/']*\d{2,4}$")  # Jan-24, March 2024

ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
DELIMITERS = ",;\t|"
HEADER_SCAN_ROWS = 15
TOTAL_WORDS = ("total", "grand total", "sub total", "subtotal", "कुल", "योग")


@dataclass
class RawTable:
    frame: pd.DataFrame
    sheet: str | None = None
    header_row: int | None = None  # 0-based row in the file; None when the file has no header
    notes: list[str] = field(default_factory=list)


class UnreadableFile(ValueError):
    """Raised with a message that can be shown to the user."""


def _decode(data: bytes) -> tuple[str, str]:
    for enc in ENCODINGS:
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise UnreadableFile("The file is not readable text. Upload a CSV or Excel file.")


def _sniff(text: str) -> str:
    sample = "\n".join(text.splitlines()[:50])
    try:
        return csv.Sniffer().sniff(sample, delimiters=DELIMITERS).delimiter
    except csv.Error:
        counts = {d: sample.count(d) for d in DELIMITERS}
        return max(counts, key=counts.get) if max(counts.values()) else ","


def _is_label(v) -> bool:
    """Header-like cell: text that is not a number or a date, or a period label as in wide sheets ('Jan-2024')."""
    if isinstance(v, (pd.Timestamp, datetime)):
        return True  # Excel stores wide-layout month headers as dates
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    s = str(v).strip()
    if not s:
        return False
    if MONTH_HEADER.match(s) or looks_like_period(s):
        return True
    return pd.isna(to_number(pd.Series([s])).iloc[0]) and date_share(pd.Series([s])) == 0


def find_header_row(grid: pd.DataFrame) -> int | None:
    """First row (within the first 15) that is wide enough and mostly text labels, followed by data."""
    head = grid.head(HEADER_SCAN_ROWS)
    filled = head.notna().sum(axis=1)
    widest = int(filled.max()) if len(filled) else 0
    if widest < 2:
        return None
    for i, row in head.iterrows():
        cells = [v for v in row if not (v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "")]
        if len(cells) < max(2, 0.6 * widest):
            continue  # title or blank row
        text_share = sum(_is_label(v) for v in cells) / len(cells)
        if text_share >= 0.7:
            return int(i)
        return None  # the first wide row is data: the file has no header
    return None


def _frame_from_grid(grid: pd.DataFrame, sheet: str | None) -> RawTable:
    notes: list[str] = []
    grid = grid.dropna(how="all").dropna(axis=1, how="all")
    grid = grid.loc[:, ~grid.apply(lambda c: c.astype(str).str.strip().eq("").all())]
    if grid.empty:
        return RawTable(pd.DataFrame(), sheet, None, ["Sheet is empty."])
    grid = grid.reset_index(drop=True)
    h = find_header_row(grid)
    if h is None:
        frame = grid.copy()
        frame.columns = [f"Column{i + 1}" for i in range(frame.shape[1])]
        notes.append("No header row found; columns named Column1, Column2, ...")
    else:
        if h > 0:
            notes.append(f"Skipped {h} title row(s) above the header.")
        names, seen = [], {}
        for i, v in enumerate(grid.iloc[h]):
            name = str(v).strip() if pd.notna(v) and str(v).strip() else f"Column{i + 1}"
            if isinstance(v, pd.Timestamp):
                name = v.strftime("%Y-%m-%d")
            seen[name] = seen.get(name, 0) + 1
            names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
        frame = grid.iloc[h + 1:].copy()
        frame.columns = names
    frame = frame.reset_index(drop=True)
    first_text = frame.apply(lambda r: next((str(v).strip().lower() for v in r if isinstance(v, str) and v.strip()), ""), axis=1)
    totals = first_text.isin(TOTAL_WORDS) | first_text.str.startswith("total")
    if totals.any():
        frame = frame[~totals].reset_index(drop=True)
        notes.append(f"Removed {int(totals.sum())} 'Total' row(s).")
    return RawTable(frame, sheet, h, notes)


def read_bytes(data: bytes, filename: str) -> list[RawTable]:
    suffix = Path(filename).suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        try:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None)
        except Exception as exc:  # corrupt or password-protected workbook
            raise UnreadableFile(f"Could not open the Excel file ({type(exc).__name__}).") from exc
        return [_frame_from_grid(g, name) for name, g in sheets.items()]
    if suffix in (".csv", ".tsv", ".txt", ""):
        text, enc = _decode(data)
        if not text.strip():
            raise UnreadableFile("The file is empty.")
        delim = "\t" if suffix == ".tsv" else _sniff(text)
        grid = pd.read_csv(io.StringIO(text), sep=delim, header=None, dtype=str, skip_blank_lines=False,
                           keep_default_na=False, na_values=[""], engine="python", on_bad_lines="skip")
        table = _frame_from_grid(grid, None)
        if enc not in ("utf-8", "utf-8-sig"):
            table.notes.append(f"Read with {enc} encoding.")
        if delim != ",":
            table.notes.append(f"Columns separated by {delim!r}.")
        return [table]
    raise UnreadableFile(f"File type {suffix or '(none)'} is not supported. Upload CSV or Excel.")


def read_path(path: str | Path) -> list[RawTable]:
    path = Path(path)
    return read_bytes(path.read_bytes(), path.name)
