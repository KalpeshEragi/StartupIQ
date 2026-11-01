"""Cap table maths: fully diluted ownership, vesting, round history and the dilution simulator. Exact, no estimates.

Conventions (stated on the page, since cap tables differ by agreement):
- Fully diluted = every issued share, with the whole ESOP pool counted; options granted are carved out of the pool,
  so a grant moves ownership from the pool to the employee without changing the total.
- Simulated rounds use the pre-money method: the pool top-up and SAFE conversions come out of the pre-money, so they
  dilute existing holders, not the new investor.
- A SAFE converts at the lower of its cap price (cap / fully diluted shares before the round) and the round price
  less its discount; with neither, at the round price.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_DOWN, Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import FundingRound, HolderKind, OptionGrant, Safe, Shareholder, ShareIssue


def months_between(start: date, end: date) -> int:
    months = (end.year - start.year) * 12 + end.month - start.month
    return months - (end.day < start.day)


def vested(grant: OptionGrant, today: date) -> int:
    elapsed = months_between(grant.granted_on, today)
    if elapsed < grant.cliff_months:
        return 0
    return min(grant.shares, grant.shares * elapsed // max(grant.vesting_months, 1))


@dataclass
class Book:
    holders: dict[uuid.UUID, Shareholder]
    issues: list[ShareIssue]
    grants: list[OptionGrant]
    safes: list[Safe]
    rounds: list[FundingRound]


def load(db: Session, startup_id: uuid.UUID) -> Book:
    def q(model):
        return list(db.scalars(select(model).where(model.startup_id == startup_id)))
    return Book({h.id: h for h in q(Shareholder)}, q(ShareIssue), q(OptionGrant), q(Safe),
                sorted(q(FundingRound), key=lambda r: r.closed_on))


def positions(book: Book, until: date | None = None) -> dict[uuid.UUID, int]:
    """Fully diluted shares per holder (optionally as of a date): issues, plus grants moved from the pool."""
    fd: dict[uuid.UUID, int] = {}
    for i in book.issues:
        if until is None or i.issued_on <= until:
            fd[i.shareholder_id] = fd.get(i.shareholder_id, 0) + i.shares
    pool = next((h.id for h in book.holders.values() if h.kind == HolderKind.esop_pool), None)
    for g in book.grants:
        if until is None or g.granted_on <= until:
            fd[g.shareholder_id] = fd.get(g.shareholder_id, 0) + g.shares
            if pool is not None:
                fd[pool] = fd.get(pool, 0) - g.shares
    return fd


def cap_table(book: Book, today: date) -> dict:
    fd = positions(book)
    total = sum(fd.values())
    issued = {h: 0 for h in book.holders}
    for i in book.issues:
        issued[i.shareholder_id] += i.shares
    options = {h: sum(g.shares for g in book.grants if g.shareholder_id == h) for h in book.holders}
    vest = {h: sum(vested(g, today) for g in book.grants if g.shareholder_id == h) for h in book.holders}
    invested = {h: Decimal(0) for h in book.holders}
    for i in book.issues:
        if i.price_per_share is not None and book.holders[i.shareholder_id].kind != HolderKind.esop_pool:
            invested[i.shareholder_id] += i.price_per_share * i.shares
    rows = [{"id": str(h.id), "name": h.name, "kind": str(h.kind), "email": h.email, "shares": issued[h.id],
             "options": options[h.id], "vested_options": vest[h.id], "fully_diluted": fd.get(h.id, 0),
             "ownership": fd.get(h.id, 0) / total if total else 0.0, "invested": float(invested[h.id])}
            for h in book.holders.values()]
    rows.sort(key=lambda r: -r["fully_diluted"])
    history = []
    for r in book.rounds:
        after = positions(book, r.closed_on)
        t = sum(after.values())
        founders = sum(v for k, v in after.items() if book.holders[k].kind == HolderKind.founder)
        history.append({"id": str(r.id), "name": r.name, "closed_on": r.closed_on.isoformat(), "pre_money": float(r.pre_money),
                        "amount_raised": float(r.amount_raised), "post_money": float(r.pre_money + r.amount_raised),
                        "price_per_share": float(r.price_per_share), "founders_after": founders / t if t else 0.0})
    return {
        "total_fully_diluted": total, "holders": rows, "rounds": history,
        "safes": [{"id": str(s.id), "holder": book.holders[s.shareholder_id].name, "amount": float(s.amount),
                   "valuation_cap": float(s.valuation_cap) if s.valuation_cap is not None else None,
                   "discount": float(s.discount) if s.discount is not None else None, "signed_on": s.signed_on.isoformat(),
                   "converted": s.converted_round_id is not None} for s in book.safes],
        "grants": [{"id": str(g.id), "holder": book.holders[g.shareholder_id].name, "shares": g.shares,
                    "granted_on": g.granted_on.isoformat(), "vesting_months": g.vesting_months, "cliff_months": g.cliff_months,
                    "vested": vested(g, today)} for g in book.grants],
    }


@dataclass
class Simulation:
    price: Decimal
    new_shares: int
    pool_top_up: int
    safe_shares: dict[uuid.UUID, int]   # safe id -> shares
    before: dict[uuid.UUID, int]
    pre_total: int


def simulate(book: Book, new_money: Decimal, pre_money: Decimal, pool_target: Decimal) -> Simulation:
    """Price a round. pool_target is the share of the post-round fully diluted total the unallocated pool should be."""
    before = positions(book)
    pre_total = sum(before.values())
    if pre_total <= 0:
        raise ValueError("Set up the cap table first: there are no shares yet")
    pool = next((h.id for h in book.holders.values() if h.kind == HolderKind.esop_pool), None)
    pool_now = before.get(pool, 0) if pool else 0
    open_safes = [s for s in book.safes if s.converted_round_id is None]
    price = pre_money / pre_total
    pool_add, safe_shares, new_shares = 0, {}, 0
    for _ in range(200):  # the price depends on the top-up and conversions, which depend on the price: iterate to a fixed point
        safe_shares = {}
        for s in open_safes:
            candidates = [price * (1 - (s.discount or 0))]
            if s.valuation_cap:
                candidates.append(s.valuation_cap / pre_total)
            safe_shares[s.id] = int((s.amount / min(candidates)).to_integral_value(ROUND_DOWN))
        new_shares = int((new_money / price).to_integral_value(ROUND_DOWN))
        post_total = pre_total + pool_add + sum(safe_shares.values()) + new_shares
        target = int((pool_target * post_total).to_integral_value(ROUND_DOWN))
        next_pool_add = max(0, target - pool_now)
        next_price = pre_money / (pre_total + next_pool_add + sum(safe_shares.values()))
        if next_pool_add == pool_add and abs(next_price - price) < Decimal("0.000001"):
            break
        pool_add, price = next_pool_add, next_price
    new_shares = int((new_money / price).to_integral_value(ROUND_DOWN))
    return Simulation(price.quantize(Decimal("0.000001")), new_shares, pool_add, safe_shares, before, pre_total)


def simulation_table(book: Book, sim: Simulation, investor: str) -> list[dict]:
    after = dict(sim.before)
    pool = next((h.id for h in book.holders.values() if h.kind == HolderKind.esop_pool), None)
    if sim.pool_top_up:
        after[pool or "new_pool"] = after.get(pool or "new_pool", 0) + sim.pool_top_up
    for s in book.safes:
        if s.id in sim.safe_shares:
            after[s.shareholder_id] = after.get(s.shareholder_id, 0) + sim.safe_shares[s.id]
    after["new_investor"] = sim.new_shares
    t_before, t_after = sim.pre_total, sum(after.values())
    names = {k: (h.name, str(h.kind)) for k, h in book.holders.items()}
    names["new_pool"], names["new_investor"] = ("ESOP pool (new)", "esop_pool"), (investor or "New investor", "investor")
    rows = [{"name": names[k][0], "kind": names[k][1], "before": sim.before.get(k, 0), "after": v,
             "ownership_before": sim.before.get(k, 0) / t_before if t_before else 0.0, "ownership_after": v / t_after}
            for k, v in after.items()]
    return sorted(rows, key=lambda r: -r["after"])
