"""Unit economics from the founder's few monthly numbers plus the P&L. Every ratio is None when an input is missing."""
from __future__ import annotations

from decimal import Decimal

from app.finance.pnl import Month
from app.models import UnitMetric


def _div(a, b) -> float | None:
    try:
        return float(Decimal(a) / Decimal(b)) if a is not None and b not in (None, 0) else None
    except (ArithmeticError, TypeError, ValueError):
        return None


def physical(m: Month, u: UnitMetric | None) -> dict:
    u = u or UnitMetric()
    return {
        "units_sold": u.units_sold, "orders": u.orders,
        "inventory_value": float(u.inventory_value) if u.inventory_value is not None else None,
        "average_order_value": _div(m.revenue, u.orders),
        "revenue_per_unit": _div(m.revenue, u.units_sold),
        "cogs_per_unit": _div(m.cogs, u.units_sold) if m.cogs else None,
        # months of stock at this month's cost of goods
        "inventory_months": _div(u.inventory_value, m.cogs) if m.cogs else None,
    }


def digital(m: Month, u: UnitMetric | None, previous: UnitMetric | None) -> dict:
    u = u or UnitMetric()
    start = previous.customers if previous is not None and previous.customers is not None else None
    churn = _div(u.churned_customers, start)
    arpu = _div(m.revenue, u.customers)
    margin = _div(m.gross_profit, m.revenue)
    cac = _div(m.opex.get("marketing"), u.new_customers)
    ltv = arpu * margin / churn if arpu is not None and margin is not None and churn else None
    return {
        "customers": u.customers, "new_customers": u.new_customers, "churned_customers": u.churned_customers,
        "mrr": float(u.mrr) if u.mrr is not None else None, "churn_rate": churn, "arpu": arpu, "cac": cac,
        "ltv": ltv, "ltv_to_cac": _div(ltv, cac) if ltv is not None else None,
        "payback_months": _div(cac, arpu * margin) if cac is not None and arpu and margin else None,
    }
