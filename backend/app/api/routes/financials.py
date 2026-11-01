"""Monthly P&L, dashboard KPIs and unit economics. Everything is computed from transactions on request, so a
correction in the review inbox shows up in the numbers at once."""
import uuid
from datetime import date
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import Access, startup_access
from app.core.db import get_db
from app.finance import units
from app.finance.pnl import Month, build_months, label, runway
from app.models import BankAccount, ProductType, UnitMetric
from app.services import audit

router = APIRouter(prefix="/startups/{startup_id}", tags=["financials"])
MONTH = Path(pattern=r"^\d{4}-(0[1-9]|1[0-2])$", description="YYYY-MM")


def _f(v: Decimal | None) -> float | None:
    return None if v is None else round(float(v), 2)


def _month_out(m: Month) -> dict:
    expenses = m.cogs + m.opex_total
    return {
        "month": m.month.isoformat()[:7], "revenue": _f(m.revenue), "other_income": _f(m.other_income), "cogs": _f(m.cogs),
        "gross_profit": _f(m.gross_profit), "gross_margin": float(m.gross_profit / m.revenue) if m.revenue else None,
        "opex": sorted(({"key": k, "label": label(k), "amount": _f(v)} for k, v in m.opex.items() if v),
                       key=lambda x: -x["amount"]),
        "opex_total": _f(m.opex_total), "expenses": _f(expenses), "ebitda": _f(m.ebitda),
        "cash_in": _f(m.cash_in), "cash_out": _f(m.cash_out), "net_cash_flow": _f(m.net_cash_flow), "burn": _f(m.burn),
        "financing_in": _f(m.financing_in), "taxes_paid": _f(m.taxes_paid), "loan_repaid": _f(m.loan_repaid),
        "closing_cash": _f(m.closing_cash), "lines": m.lines, "needs_review": m.needs_review,
        "uncategorised": _f(m.uncategorised),
    }


def _runway_out(r: dict) -> dict:
    months = None if r["months"] is None else round(r["months"], 1)
    return {"cash": _f(r["cash"]), "avg_burn": _f(r["avg_burn"]), "months": months, "status": r["status"]}


@router.get("/financials/pnl")
def pnl(access: Access = Depends(startup_access()), db: Session = Depends(get_db)) -> dict:
    months = build_months(db, access.startup.id)
    return {"currency": access.startup.currency, "months": [_month_out(m) for m in months], "runway": _runway_out(runway(months))}


def _metrics(db: Session, startup_id: uuid.UUID) -> dict[date, UnitMetric]:
    return {u.month: u for u in db.scalars(select(UnitMetric).where(UnitMetric.startup_id == startup_id))}


def _units(product_type: ProductType, m: Month, metrics: dict[date, UnitMetric]) -> dict:
    if product_type == ProductType.physical:
        return units.physical(m, metrics.get(m.month))
    prev = date(m.month.year - (m.month.month == 1), (m.month.month - 2) % 12 + 1, 1)
    return units.digital(m, metrics.get(m.month), metrics.get(prev))


@router.get("/financials/dashboard")
def dashboard(access: Access = Depends(startup_access()), db: Session = Depends(get_db)) -> dict:
    months = build_months(db, access.startup.id)
    with_data = [m for m in months if m.lines]
    if not with_data:
        return {"currency": access.startup.currency, "has_data": False}
    latest = with_data[-1]
    prev = with_data[-2] if len(with_data) > 1 else None
    metrics = _metrics(db, access.startup.id)
    return {
        "currency": access.startup.currency, "has_data": True, "month": latest.month.isoformat()[:7],
        "kpis": {
            "revenue": _f(latest.revenue),
            "revenue_growth": float(latest.revenue / prev.revenue - 1) if prev and prev.revenue else None,
            "gross_margin": float(latest.gross_profit / latest.revenue) if latest.revenue else None,
            "ebitda": _f(latest.ebitda), "burn": _f(latest.burn),
        },
        "runway": _runway_out(runway(months)),
        "needs_review": sum(m.needs_review for m in months),
        "uncategorised": _f(latest.uncategorised),
        "series": [{"month": m.month.isoformat()[:7], "revenue": _f(m.revenue), "expenses": _f(m.cogs + m.opex_total),
                    "ebitda": _f(m.ebitda), "closing_cash": _f(m.closing_cash)} for m in months[-12:]],
        "top_expenses": _month_out(latest)["opex"][:6],
        "units": _units(access.startup.product_type, latest, metrics),
        "units_entered": latest.month in metrics,
    }


# Upper bounds keep inputs inside the database columns (Integer counts, Numeric(16, 2) money) on every database.
MAX_COUNT = 2_000_000_000
MAX_MONEY = 1e13


class UnitMetricIn(BaseModel):
    units_sold: int | None = Field(default=None, ge=0, le=MAX_COUNT)
    orders: int | None = Field(default=None, ge=0, le=MAX_COUNT)
    inventory_value: float | None = Field(default=None, ge=0, le=MAX_MONEY)
    customers: int | None = Field(default=None, ge=0, le=MAX_COUNT)
    new_customers: int | None = Field(default=None, ge=0, le=MAX_COUNT)
    churned_customers: int | None = Field(default=None, ge=0, le=MAX_COUNT)
    mrr: float | None = Field(default=None, ge=0, le=MAX_MONEY)


FIELDS = list(UnitMetricIn.model_fields)


@router.get("/unit-metrics")
def list_unit_metrics(access: Access = Depends(startup_access()), db: Session = Depends(get_db)) -> list[dict]:
    months = {m.month: m for m in build_months(db, access.startup.id)}
    metrics = _metrics(db, access.startup.id)
    out = []
    for month in sorted(set(months) | set(metrics), reverse=True):
        u = metrics.get(month)
        values = {f: getattr(u, f) if u else None for f in FIELDS}
        row = {"month": month.isoformat()[:7], **{f: float(v) if isinstance(v, Decimal) else v for f, v in values.items()}}
        if month in months:
            row["computed"] = _units(access.startup.product_type, months[month], metrics)
        out.append(row)
    return out


@router.put("/unit-metrics/{month}")
def save_unit_metrics(body: UnitMetricIn, request: Request, month: str = MONTH, access: Access = Depends(startup_access()),
                      db: Session = Depends(get_db)) -> dict:
    first = date(int(month[:4]), int(month[5:]), 1)
    if first > date.today():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "That month has not happened yet")
    row = db.scalar(select(UnitMetric).where(UnitMetric.startup_id == access.startup.id, UnitMetric.month == first))
    if row is None:
        row = UnitMetric(startup_id=access.startup.id, month=first)
        db.add(row)
    for k, v in body.model_dump().items():
        setattr(row, k, v)
    audit.record(db, "unit_metrics.save", user_id=access.user.id, startup_id=access.startup.id, entity_id=row.id,
                 details={"month": month}, request=request)
    db.commit()
    return {"month": month, **body.model_dump()}


class OpeningBalanceIn(BaseModel):
    opening_balance: float | None = Field(ge=-MAX_MONEY, le=MAX_MONEY, allow_inf_nan=False)


@router.patch("/bank-accounts/{account_id}")
def set_opening_balance(account_id: uuid.UUID, body: OpeningBalanceIn, access: Access = Depends(startup_access()),
                        db: Session = Depends(get_db)) -> dict:
    account = db.get(BankAccount, account_id)
    if account is None or account.startup_id != access.startup.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Bank account not found")
    account.opening_balance = None if body.opening_balance is None else Decimal(str(body.opening_balance))
    db.commit()
    return {"id": str(account.id), "opening_balance": body.opening_balance}
