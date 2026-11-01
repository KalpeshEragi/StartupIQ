"""Checks that make an uploaded statement trustworthy, in plain language for the founder.

What these checks can and cannot show: they catch edited, missing or reordered lines and statements missing between
uploads, because a bank's running balance has to add up. They cannot prove a file came from the bank: someone can edit
a CSV consistently before uploading it. Only bank-signed data (signed PDF statements, Account Aggregator) proves that.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from app.categorisation.statements import ParsedRow

MAX_ISSUES = 20  # per statement; the count says how many more there were


def file_sha256(data: bytes) -> str:
    """The file's fingerprint: any change to the file, even one digit, gives a different value."""
    return hashlib.sha256(data).hexdigest()


def rows_seal(rows: Sequence[tuple]) -> str:
    """One fingerprint over an import's stored lines: (fingerprint, date, direction, amount, balance, description).
    Recomputed on every report, so a line edited or deleted directly in the database is caught. Categories are left out
    on purpose: the founder changes those, and every change is in the audit log."""
    def line(r: tuple) -> str:
        fp, when, direction, amount, balance, desc = r
        return f"{fp}|{when}|{direction}|{Decimal(amount):.2f}|{'' if balance is None else f'{Decimal(balance):.2f}'}|{desc}"
    return hashlib.sha256("\n".join(sorted(map(line, rows))).encode()).hexdigest()


def inr(amount: Decimal | float | None) -> str:
    """₹12,34,567.89 (Indian grouping)."""
    if amount is None:
        return "-"
    sign = "-" if amount < 0 else ""
    whole, frac = f"{abs(Decimal(amount)):.2f}".split(".")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return f"{sign}₹{','.join([*groups, tail])}.{frac}"


@dataclass
class BalanceCheck:
    status: str                      # verified | warnings | unverified
    checked: int = 0                 # consecutive line pairs whose balances could be compared
    breaks: int = 0
    opening: Decimal | None = None   # balance before the earliest line
    closing: Decimal | None = None   # balance after the latest line
    issues: list[str] = field(default_factory=list)


def _signed(row: ParsedRow) -> Decimal:
    return row.amount if row.direction == "credit" else -row.amount


def check_balances(rows: Sequence[ParsedRow]) -> BalanceCheck:
    """Every line: previous balance + money in - money out = this balance. Banks export oldest-first or newest-first;
    the order that adds up best is used."""
    pairs = [i for i in range(1, len(rows)) if rows[i].balance is not None and rows[i - 1].balance is not None]
    if not any(r.balance is not None for r in rows):
        return BalanceCheck("unverified", issues=["The file has no balance column, so its lines cannot be checked "
                                                  "against each other."])
    def oldest_first_ok(i: int) -> bool:
        return rows[i - 1].balance + _signed(rows[i]) == rows[i].balance  # type: ignore[operator]

    def newest_first_ok(i: int) -> bool:
        return rows[i].balance + _signed(rows[i - 1]) == rows[i - 1].balance  # type: ignore[operator]

    newest_first = sum(map(newest_first_ok, pairs)) > sum(map(oldest_first_ok, pairs))  # oldest-first when tied
    ok = newest_first_ok if newest_first else oldest_first_ok
    broken = [i for i in pairs if not ok(i)]

    issues: list[str] = []
    for i in broken[:MAX_ISSUES]:
        later, earlier = (rows[i - 1], rows[i]) if newest_first else (rows[i], rows[i - 1])
        expected = earlier.balance + _signed(later)  # type: ignore[operator]
        issues.append(f"Line {later.line} ({later.txn_date:%d %b %Y}, {later.description[:60]}): the balance should be "
                      f"{inr(expected)} but the file says {inr(later.balance)}. A line may have been edited or is missing.")
    if len(broken) > MAX_ISSUES:
        issues.append(f"…and {len(broken) - MAX_ISSUES} more lines that do not add up.")
    missing = sum(r.balance is None for r in rows)
    if missing:
        issues.append(f"{missing} line(s) have no balance and could not be checked.")

    chrono = list(reversed(rows)) if newest_first else list(rows)
    first = next((r for r in chrono if r.balance is not None), None)
    last = next((r for r in reversed(chrono) if r.balance is not None), None)
    opening = first.balance - _signed(first) if first is not None and first is chrono[0] else None  # type: ignore[operator]
    closing = last.balance if last is not None and last is chrono[-1] else None
    if not pairs:  # a single line still gives opening and closing balances for the gap check between statements
        return BalanceCheck("unverified", 0, 0, opening, closing, ["Too few lines with a balance to check."])
    return BalanceCheck("warnings" if broken else "verified", len(pairs), len(broken), opening, closing, issues)


@dataclass
class StatementSpan:
    """What continuity needs from one stored import."""
    filename: str
    period_start: date | None
    period_end: date | None
    opening: Decimal | None
    closing: Decimal | None


def continuity(spans: Sequence[StatementSpan]) -> list[str]:
    """Statements of one account, side by side: each should start where the one before it ended. A jump means a
    statement (or lines) between them is missing. Overlapping uploads are skipped: duplicates are already ignored."""
    ordered = sorted((s for s in spans if s.period_start and s.period_end), key=lambda s: (s.period_start, s.period_end))
    issues = []
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        if cur.period_start < prev.period_end or prev.closing is None or cur.opening is None:  # type: ignore[operator]
            continue
        if prev.closing != cur.opening:
            issues.append(f"{prev.filename} ends at {inr(prev.closing)} on {prev.period_end:%d %b %Y}, but {cur.filename} "
                          f"starts at {inr(cur.opening)} on {cur.period_start:%d %b %Y}. A statement or some lines "
                          f"between them may be missing.")
    return issues
