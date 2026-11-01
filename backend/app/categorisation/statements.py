"""Read a bank statement export (CSV or Excel) into clean rows.

Handles what Indian bank exports actually look like: title and address lines above the table, column names that
differ by bank (Narration / Particulars / Description, Withdrawal Amt / Debit / Dr), either separate debit and credit
columns or one amount with a Dr/Cr column, ₹ signs and thousands separators, and day-first dates.
"""
from __future__ import annotations

import csv
import hashlib
import io
import itertools
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation

import pandas as pd

from app.categorisation.text import clean

MAX_ROWS = 50_000
HEADER_SCAN = 40  # rows searched for the header line

# Normalised header name fragments per column role; earlier fragments are better matches.
ROLES: dict[str, tuple[str, ...]] = {
    "date": ("txndate", "transactiondate", "trandate", "postingdate", "postdate", "date", "valuedate", "valuedt"),
    "description": ("narration", "description", "particulars", "transactiondetails", "details", "remarks", "transactionremarks"),
    "debit": ("withdrawalamt", "withdrawalamount", "withdrawal", "withdrawals", "debitamount", "debit", "dr", "paidout"),
    "credit": ("depositamt", "depositamount", "deposit", "deposits", "creditamount", "credit", "cr", "paidin"),
    "amount": ("transactionamount", "amount", "amt"),
    "type": ("crdr", "drcr", "type", "transactiontype", "debitcredit"),
    "balance": ("closingbalance", "availablebalance", "balance", "runningbalance"),
}


class StatementError(ValueError):
    """The file cannot be read as a bank statement; the message is shown to the founder."""


@dataclass
class ParsedRow:
    txn_date: date
    description: str
    direction: str  # "debit" | "credit"
    amount: Decimal
    balance: Decimal | None
    fingerprint: str = ""
    vendor_key: str = ""


@dataclass
class ParsedStatement:
    rows: list[ParsedRow]
    columns: dict[str, str]           # role -> header found in the file
    skipped: int = 0                  # rows with an unreadable date or amount
    notes: list[str] = field(default_factory=list)


def _norm(name: object) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower())


def _match_columns(headers: list[object]) -> dict[str, int]:
    """role -> column index, best fragment first, each column used once."""
    found: dict[str, int] = {}
    used: set[int] = set()
    normed = [_norm(h) for h in headers]
    for role, fragments in ROLES.items():
        for fragment in fragments:
            exact = [i for i, n in enumerate(normed) if n == fragment and i not in used]
            partial = [i for i, n in enumerate(normed) if fragment in n and i not in used and len(fragment) >= 4]
            hit = (exact or partial or [None])[0]
            if hit is not None:
                found[role], _ = hit, used.add(hit)
                break
    return found


def _usable(found: dict[str, int]) -> bool:
    has_amount = ("debit" in found and "credit" in found) or "amount" in found
    return "date" in found and "description" in found and has_amount


def _read_frames(data: bytes, filename: str) -> list[pd.DataFrame]:
    name = filename.lower()
    try:
        if name.endswith((".xlsx", ".xlsm", ".xls")):
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, dtype=str, nrows=MAX_ROWS + HEADER_SCAN)
            return list(sheets.values())
        text = data.decode("utf-8-sig", errors="replace")
        sep = max([",", ";", "\t", "|"], key=lambda s: text[:5000].count(s))
        # Read row by row: title lines above the table are shorter than the table, and a strict CSV reader would take
        # the first line's width and drop every real row as malformed.
        rows = list(itertools.islice(csv.reader(io.StringIO(text), delimiter=sep), MAX_ROWS + HEADER_SCAN))
        width = max((len(r) for r in rows), default=0)
        return [pd.DataFrame([r + [None] * (width - len(r)) for r in rows], dtype=object)]
    except (ValueError, UnicodeError, csv.Error, pd.errors.ParserError) as exc:
        raise StatementError("We could not open this file. Upload the CSV or Excel statement downloaded from your bank.") from exc


def _money(value: object) -> Decimal | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().upper().replace("₹", "").replace("INR", "").replace("RS.", "").replace(",", "").strip()
    s = re.sub(r"\s*(CR|DR)\.?$", "", s)
    if s in ("", "-", "NAN", "NONE"):
        return None
    negative = s.startswith("(") and s.endswith(")")
    try:
        d = Decimal(s.strip("()"))
    except InvalidOperation:
        return None
    return -d if negative else d


def _date(value: object) -> date | None:
    if value is None or str(value).strip() in ("", "nan"):
        return None
    s = str(value).strip()
    iso = re.match(r"^\d{4}-\d{2}-\d{2}", s)
    parsed = pd.to_datetime(s[:10] if iso else s, dayfirst=not iso, errors="coerce")
    return None if pd.isna(parsed) else parsed.date()


def parse_statement(data: bytes, filename: str) -> ParsedStatement:
    if not data:
        raise StatementError("The file is empty.")
    for frame in _read_frames(data, filename):
        frame = frame.dropna(how="all")
        for header_row in range(min(HEADER_SCAN, len(frame))):
            found = _match_columns(list(frame.iloc[header_row]))
            if _usable(found):
                body = frame.iloc[header_row + 1:]
                headers = list(frame.iloc[header_row])
                return _rows(body, found, {role: str(headers[i]) for role, i in found.items()}, header_row)
    raise StatementError("We could not find the date, description and amount columns. Upload the statement as your bank "
                         "exports it, with its header row (for example: Date, Narration, Withdrawal, Deposit, Balance).")


def vendor_key(description: str) -> str:
    """What the founder's memory matches on: the cleaned description without leftover number tokens."""
    return " ".join(t for t in clean(description).split() if t != "0")[:200]


def _rows(body: pd.DataFrame, cols: dict[str, int], names: dict[str, str], header_row: int) -> ParsedStatement:
    rows: list[ParsedRow] = []
    skipped = 0
    seen: Counter[tuple] = Counter()
    for values in body.itertuples(index=False):
        def get(role: str, values: tuple = values) -> object:
            return values[cols[role]] if role in cols else None
        when = _date(get("date"))
        desc = " ".join(str(get("description") or "").split())[:500]
        if when is None or not desc or desc.lower() == "nan":
            if any(str(v).strip() not in ("", "nan", "None") for v in values):
                skipped += 1  # a real-looking line we cannot read; totals and footer lines also land here
            continue
        if "debit" in cols and "credit" in cols:
            debit, credit = _money(get("debit")), _money(get("credit"))
            if debit and debit != 0:
                direction, amount = "debit", abs(debit)
            elif credit and credit != 0:
                direction, amount = "credit", abs(credit)
            else:
                skipped += 1
                continue
        else:
            amount = _money(get("amount"))
            if amount is None or amount == 0:
                skipped += 1
                continue
            kind = _norm(get("type")) if "type" in cols else ""
            if kind.startswith(("cr", "credit", "c")) and kind not in ("", "nan"):
                direction = "credit"
            elif kind.startswith(("dr", "debit", "d")):
                direction = "debit"
            else:
                direction = "debit" if amount < 0 else "credit"
            amount = abs(amount)
        balance = _money(get("balance"))
        key = (when, direction, amount, desc.lower())
        seen[key] += 1  # identical lines on one statement are separate payments; the counter keeps them apart
        fingerprint = hashlib.sha256(f"{when}|{direction}|{amount:.2f}|{desc.lower()}|{seen[key]}".encode()).hexdigest()
        rows.append(ParsedRow(when, desc, direction, amount.quantize(Decimal("0.01")), balance, fingerprint, vendor_key(desc)))
    if not rows:
        raise StatementError("No transactions could be read from this file.")
    notes = [f"Skipped {header_row} line(s) above the table."] if header_row else []
    if skipped:
        notes.append(f"{skipped} line(s) had no readable date or amount and were left out (totals and footers usually).")
    return ParsedStatement(rows=rows, columns=names, skipped=skipped, notes=notes)
