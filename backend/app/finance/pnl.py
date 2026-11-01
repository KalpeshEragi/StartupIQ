"""Monthly P&L, cash and runway, built from categorised transactions. Exact arithmetic, no estimates.

How each category counts:
- Revenue                  revenue (customer receipts)
- Other income             interest, dividends and other income (below EBITDA)
- Cost of goods            cogs (founder-assigned)
- Operating expenses       every expense category, bank charges, unconfirmed payments to people (as people costs)
                           and debits not categorised yet (shown separately, so the founder sees what is missing)
- Not in the P&L           GST/TDS payments, loan principal (EMIs), money raised, cash moves, refunds, failed lines:
                           they move cash but are not income or expense, so they appear only in the cash flow
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.categorisation.categories import ALL_CATEGORIES, EXPENSE_CATEGORIES
from app.models import BankAccount, Transaction

ZERO = Decimal("0")
OPEX = set(EXPENSE_CATEGORIES) | {"bank_charges"}
NOT_IN_PNL = {"taxes", "loan_repayment", "financing", "cash", "refund"}
UNCATEGORISED = "uncategorised"


def month_start(d: date) -> date:
    return d.replace(day=1)


@dataclass
class Month:
    month: date
    revenue: Decimal = ZERO
    other_income: Decimal = ZERO
    cogs: Decimal = ZERO
    opex: dict[str, Decimal] = field(default_factory=lambda: defaultdict(lambda: ZERO))
    cash_in: Decimal = ZERO
    cash_out: Decimal = ZERO
    financing_in: Decimal = ZERO
    taxes_paid: Decimal = ZERO
    loan_repaid: Decimal = ZERO
    closing_cash: Decimal | None = None
    lines: int = 0
    needs_review: int = 0
    uncategorised: Decimal = ZERO

    @property
    def gross_profit(self) -> Decimal:
        return self.revenue - self.cogs

    @property
    def opex_total(self) -> Decimal:
        return sum(self.opex.values(), ZERO)

    @property
    def ebitda(self) -> Decimal:
        return self.gross_profit - self.opex_total

    @property
    def net_cash_flow(self) -> Decimal:
        return self.cash_in - self.cash_out

    @property
    def burn(self) -> Decimal:
        """Operating burn: cash out minus cash in, leaving out money raised. Negative means cash-generating."""
        return self.cash_out - (self.cash_in - self.financing_in)


def build_months(db: Session, startup_id: uuid.UUID) -> list[Month]:
    """Every calendar month from the first to the last transaction, gaps included, oldest first."""
    txns = list(db.scalars(select(Transaction).where(Transaction.startup_id == startup_id)
                           .order_by(Transaction.txn_date, Transaction.created_at)))
    if not txns:
        return []
    months: dict[date, Month] = {}
    first, last = month_start(txns[0].txn_date), month_start(txns[-1].txn_date)
    m = first
    while m <= last:
        months[m] = Month(m)
        m = date(m.year + (m.month == 12), m.month % 12 + 1, 1)

    last_balance: dict[tuple[uuid.UUID, date], Decimal] = {}
    flows: dict[tuple[uuid.UUID, date], Decimal] = defaultdict(lambda: ZERO)
    for t in txns:
        mo = months[month_start(t.txn_date)]
        amount, cat, debit = Decimal(t.amount), t.category, str(t.direction) == "debit"
        mo.lines += 1
        mo.needs_review += t.needs_review
        if cat == "failed":
            continue  # never happened
        if debit:
            mo.cash_out += amount
        else:
            mo.cash_in += amount
        flows[(t.bank_account_id, mo.month)] += -amount if debit else amount
        if t.balance is not None:
            last_balance[(t.bank_account_id, mo.month)] = Decimal(t.balance)
        if cat == "financing" and not debit:
            mo.financing_in += amount
        elif cat == "taxes" and debit:
            mo.taxes_paid += amount
        elif cat == "loan_repayment" and debit:
            mo.loan_repaid += amount
        elif cat in NOT_IN_PNL:
            continue
        elif cat == "revenue" and not debit:
            mo.revenue += amount
        elif cat == "other_income" and not debit:
            mo.other_income += amount
        elif cat == "cogs":
            mo.cogs += amount if debit else -amount
        elif cat in OPEX or cat == "person_payment":
            key = "people" if cat == "person_payment" else cat
            mo.opex[key] += amount if debit else -amount
        elif cat is None and debit:
            mo.opex[UNCATEGORISED] += amount
            mo.uncategorised += amount
        elif cat is None:
            mo.uncategorised += amount  # money in we could not recognise: not counted as revenue until confirmed

    accounts = list(db.scalars(select(BankAccount).where(BankAccount.startup_id == startup_id)))
    for acc in accounts:
        running: Decimal | None = Decimal(acc.opening_balance) if acc.opening_balance is not None else None
        for mo in months.values():
            key = (acc.id, mo.month)
            if key in last_balance:
                running = last_balance[key]          # the bank's own balance wins
            elif running is not None:
                running += flows.get(key, ZERO)       # no balance column: opening balance plus the month's flows
            if running is not None:
                mo.closing_cash = (mo.closing_cash or ZERO) + running
    return list(months.values())


def runway(months: list[Month], window: int = 3) -> dict:
    """Months of cash left at the average operating burn of the last `window` months with data."""
    recent = [m for m in months if m.lines][-window:]
    cash = next((m.closing_cash for m in reversed(months) if m.closing_cash is not None), None)
    if not recent:
        return {"cash": None, "avg_burn": None, "months": None, "status": "no_data"}
    avg_burn = sum((m.burn for m in recent), ZERO) / len(recent)
    if cash is None:
        return {"cash": None, "avg_burn": avg_burn, "months": None, "status": "no_balance"}
    if avg_burn <= 0:
        return {"cash": cash, "avg_burn": avg_burn, "months": None, "status": "not_burning"}
    return {"cash": cash, "avg_burn": avg_burn, "months": float(cash / avg_burn), "status": "burning"}


def label(key: str) -> str:
    return "Not categorised yet" if key == UNCATEGORISED else ALL_CATEGORIES.get(key, key)
