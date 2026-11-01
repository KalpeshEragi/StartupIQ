"""Cap table (shareholders, share issues, SAFEs, option grants, priced rounds), the dilution simulator and the
fundraising pipeline. Every route is scoped to one startup and audited."""
import uuid
from datetime import date
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import Access, startup_access
from app.core.db import get_db
from app.finance import equity
from app.models import Deal, DealStage, FundingRound, HolderKind, OptionGrant, Safe, Shareholder, ShareIssue
from app.services import audit

router = APIRouter(prefix="/startups/{startup_id}", tags=["equity"])


def _own(db: Session, model, obj_id: uuid.UUID, access: Access):
    obj = db.get(model, obj_id)
    if obj is None or obj.startup_id != access.startup.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{model.__name__} not found")
    return obj


def _holder(db: Session, access: Access, holder_id: uuid.UUID | None, name: str | None, kind: HolderKind) -> Shareholder:
    """An existing shareholder by id, or by name (created if new)."""
    if holder_id is not None:
        return _own(db, Shareholder, holder_id, access)
    name = (name or "").strip()
    if not name:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Give a shareholder or a name")
    found = db.scalar(select(Shareholder).where(Shareholder.startup_id == access.startup.id, Shareholder.name == name))
    if found is None:
        found = Shareholder(startup_id=access.startup.id, name=name, kind=kind)
        db.add(found)
        db.flush()
    return found


def _pool(db: Session, access: Access) -> Shareholder:
    pool = db.scalar(select(Shareholder).where(Shareholder.startup_id == access.startup.id,
                                               Shareholder.kind == HolderKind.esop_pool))
    if pool is None:
        pool = Shareholder(startup_id=access.startup.id, name="ESOP pool", kind=HolderKind.esop_pool)
        db.add(pool)
        db.flush()
    return pool


def _log(db: Session, access: Access, action: str, entity_id: uuid.UUID | None, request: Request, **details) -> None:
    audit.record(db, action, user_id=access.user.id, startup_id=access.startup.id, entity_id=entity_id, details=details,
                 request=request)


@router.get("/cap-table")
def get_cap_table(access: Access = Depends(startup_access()), db: Session = Depends(get_db)) -> dict:
    return equity.cap_table(equity.load(db, access.startup.id), date.today())


class FounderIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    shares: int = Field(gt=0)
    email: EmailStr | None = None


class SetupIn(BaseModel):
    incorporated_on: date
    founders: list[FounderIn] = Field(min_length=1, max_length=10)
    pool_shares: int = Field(default=0, ge=0)


@router.post("/cap-table/setup", status_code=status.HTTP_201_CREATED)
def setup(body: SetupIn, request: Request, access: Access = Depends(startup_access()), db: Session = Depends(get_db)) -> dict:
    """Founders' shares (and an optional ESOP pool) at incorporation. Only for an empty cap table."""
    if db.scalar(select(ShareIssue.id).where(ShareIssue.startup_id == access.startup.id).limit(1)):
        raise HTTPException(status.HTTP_409_CONFLICT, "The cap table already has shares; add holders or rounds instead")
    for f in body.founders:
        h = Shareholder(startup_id=access.startup.id, name=f.name.strip(), kind=HolderKind.founder, email=f.email)
        db.add(h)
        db.flush()
        db.add(ShareIssue(startup_id=access.startup.id, shareholder_id=h.id, shares=f.shares, issued_on=body.incorporated_on))
    if body.pool_shares:
        db.add(ShareIssue(startup_id=access.startup.id, shareholder_id=_pool(db, access).id, share_class="pool",
                          shares=body.pool_shares, issued_on=body.incorporated_on))
    _log(db, access, "cap_table.setup", None, request, founders=len(body.founders), pool=body.pool_shares)
    db.commit()
    return get_cap_table(access, db)


class HolderIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    kind: HolderKind
    email: EmailStr | None = None


@router.post("/shareholders", status_code=status.HTTP_201_CREATED)
def add_holder(body: HolderIn, request: Request, access: Access = Depends(startup_access()),
               db: Session = Depends(get_db)) -> dict:
    h = Shareholder(startup_id=access.startup.id, name=body.name.strip(), kind=body.kind, email=body.email)
    db.add(h)
    db.flush()
    _log(db, access, "shareholder.create", h.id, request, name=h.name)
    db.commit()
    return {"id": str(h.id), "name": h.name, "kind": str(h.kind)}


@router.delete("/shareholders/{holder_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_holder(holder_id: uuid.UUID, request: Request, access: Access = Depends(startup_access()),
                  db: Session = Depends(get_db)) -> None:
    h = _own(db, Shareholder, holder_id, access)
    _log(db, access, "shareholder.delete", h.id, request, name=h.name)
    db.delete(h)
    db.commit()


class IssueIn(BaseModel):
    shareholder_id: uuid.UUID | None = None
    name: str | None = None
    kind: HolderKind = HolderKind.other
    shares: int = Field(gt=0)
    price_per_share: float | None = Field(default=None, ge=0)
    issued_on: date


@router.post("/share-issues", status_code=status.HTTP_201_CREATED)
def add_issue(body: IssueIn, request: Request, access: Access = Depends(startup_access()), db: Session = Depends(get_db)) -> dict:
    h = _holder(db, access, body.shareholder_id, body.name, body.kind)
    issue = ShareIssue(startup_id=access.startup.id, shareholder_id=h.id, shares=body.shares, issued_on=body.issued_on,
                       share_class="pool" if h.kind == HolderKind.esop_pool else "equity",
                       price_per_share=None if body.price_per_share is None else Decimal(str(body.price_per_share)))
    db.add(issue)
    db.flush()
    _log(db, access, "shares.issue", issue.id, request, holder=h.name, shares=body.shares)
    db.commit()
    return {"id": str(issue.id)}


class SafeIn(BaseModel):
    shareholder_id: uuid.UUID | None = None
    investor_name: str | None = None
    amount: float = Field(gt=0)
    valuation_cap: float | None = Field(default=None, gt=0)
    discount: float | None = Field(default=None, ge=0, lt=1)
    signed_on: date


@router.post("/safes", status_code=status.HTTP_201_CREATED)
def add_safe(body: SafeIn, request: Request, access: Access = Depends(startup_access()), db: Session = Depends(get_db)) -> dict:
    h = _holder(db, access, body.shareholder_id, body.investor_name, HolderKind.investor)
    s = Safe(startup_id=access.startup.id, shareholder_id=h.id, amount=Decimal(str(body.amount)), signed_on=body.signed_on,
             valuation_cap=None if body.valuation_cap is None else Decimal(str(body.valuation_cap)),
             discount=None if body.discount is None else Decimal(str(body.discount)))
    db.add(s)
    db.flush()
    _log(db, access, "safe.create", s.id, request, holder=h.name, amount=body.amount)
    db.commit()
    return {"id": str(s.id)}


class GrantIn(BaseModel):
    shareholder_id: uuid.UUID | None = None
    employee_name: str | None = None
    shares: int = Field(gt=0)
    granted_on: date
    vesting_months: int = Field(default=48, ge=1, le=120)
    cliff_months: int = Field(default=12, ge=0, le=60)

    @model_validator(mode="after")
    def _cliff_within_vesting(self):
        if self.cliff_months > self.vesting_months:
            raise ValueError("the cliff cannot be longer than the vesting period")
        return self


@router.post("/option-grants", status_code=status.HTTP_201_CREATED)
def add_grant(body: GrantIn, request: Request, access: Access = Depends(startup_access()), db: Session = Depends(get_db)) -> dict:
    book = equity.load(db, access.startup.id)
    pool = next((h for h in book.holders.values() if h.kind == HolderKind.esop_pool), None)
    available = equity.positions(book).get(pool.id, 0) if pool else 0
    if body.shares > available:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Only {available:,} options are left in the ESOP pool")
    h = _holder(db, access, body.shareholder_id, body.employee_name, HolderKind.employee)
    g = OptionGrant(startup_id=access.startup.id, shareholder_id=h.id, shares=body.shares, granted_on=body.granted_on,
                    vesting_months=body.vesting_months, cliff_months=body.cliff_months)
    db.add(g)
    db.flush()
    _log(db, access, "options.grant", g.id, request, holder=h.name, shares=body.shares)
    db.commit()
    return {"id": str(g.id)}


def _delete(model, action: str, obj_id: uuid.UUID, request: Request, access: Access, db: Session) -> None:
    obj = _own(db, model, obj_id, access)
    _log(db, access, action, obj.id, request)
    db.delete(obj)
    db.commit()


@router.delete("/share-issues/{obj_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_issue(obj_id: uuid.UUID, request: Request, access: Access = Depends(startup_access()),
                 db: Session = Depends(get_db)) -> None:
    _delete(ShareIssue, "shares.delete", obj_id, request, access, db)


@router.delete("/safes/{obj_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_safe(obj_id: uuid.UUID, request: Request, access: Access = Depends(startup_access()),
                db: Session = Depends(get_db)) -> None:
    _delete(Safe, "safe.delete", obj_id, request, access, db)


@router.delete("/option-grants/{obj_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_grant(obj_id: uuid.UUID, request: Request, access: Access = Depends(startup_access()),
                 db: Session = Depends(get_db)) -> None:
    _delete(OptionGrant, "options.delete", obj_id, request, access, db)


class RoundIn(BaseModel):
    new_money: float = Field(gt=0)
    pre_money: float = Field(gt=0)
    pool_target: float = Field(default=0, ge=0, le=0.5)  # unallocated pool as a share of the post-round total
    investor_name: str = Field(default="New investor", min_length=1, max_length=120)


def _simulate(db: Session, access: Access, body: RoundIn) -> tuple[equity.Book, equity.Simulation]:
    book = equity.load(db, access.startup.id)
    try:
        sim = equity.simulate(book, Decimal(str(body.new_money)), Decimal(str(body.pre_money)), Decimal(str(body.pool_target)))
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from None
    return book, sim


@router.post("/cap-table/simulate")
def simulate(body: RoundIn, access: Access = Depends(startup_access()), db: Session = Depends(get_db)) -> dict:
    book, sim = _simulate(db, access, body)
    return {"price_per_share": float(sim.price), "post_money": body.pre_money + body.new_money, "new_shares": sim.new_shares,
            "pool_top_up": sim.pool_top_up, "safe_conversions": [
                {"holder": book.holders[s.shareholder_id].name, "amount": float(s.amount), "shares": sim.safe_shares[s.id]}
                for s in book.safes if s.id in sim.safe_shares],
            "table": equity.simulation_table(book, sim, body.investor_name)}


class RecordRoundIn(RoundIn):
    name: str = Field(min_length=1, max_length=80)
    closed_on: date


@router.post("/rounds", status_code=status.HTTP_201_CREATED)
def record_round(body: RecordRoundIn, request: Request, access: Access = Depends(startup_access()),
                 db: Session = Depends(get_db)) -> dict:
    """Record a priced round exactly as simulated: investor shares, SAFE conversions and the pool top-up."""
    book, sim = _simulate(db, access, body)
    r = FundingRound(startup_id=access.startup.id, name=body.name.strip(), closed_on=body.closed_on,
                     pre_money=Decimal(str(body.pre_money)), amount_raised=Decimal(str(body.new_money)),
                     price_per_share=sim.price)
    db.add(r)
    db.flush()
    investor = _holder(db, access, None, body.investor_name, HolderKind.investor)
    db.add(ShareIssue(startup_id=access.startup.id, shareholder_id=investor.id, round_id=r.id, share_class="preference",
                      shares=sim.new_shares, price_per_share=sim.price, issued_on=body.closed_on))
    for s in book.safes:
        if s.id in sim.safe_shares:
            db.add(ShareIssue(startup_id=access.startup.id, shareholder_id=s.shareholder_id, round_id=r.id,
                              share_class="preference", shares=sim.safe_shares[s.id],
                              price_per_share=(s.amount / sim.safe_shares[s.id]).quantize(Decimal("0.000001")),
                              issued_on=body.closed_on))
            db.get(Safe, s.id).converted_round_id = r.id
    if sim.pool_top_up:
        db.add(ShareIssue(startup_id=access.startup.id, shareholder_id=_pool(db, access).id, round_id=r.id, share_class="pool",
                          shares=sim.pool_top_up, issued_on=body.closed_on))
    _log(db, access, "round.record", r.id, request, name=r.name, raised=body.new_money, pre_money=body.pre_money)
    db.commit()
    return get_cap_table(access, db)


@router.delete("/rounds/{round_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_round(round_id: uuid.UUID, request: Request, access: Access = Depends(startup_access()),
                 db: Session = Depends(get_db)) -> None:
    """Undo a recorded round: its shares go, and the SAFEs it converted are open again."""
    r = _own(db, FundingRound, round_id, access)
    for i in db.scalars(select(ShareIssue).where(ShareIssue.round_id == r.id)):
        db.delete(i)
    for s in db.scalars(select(Safe).where(Safe.converted_round_id == r.id)):
        s.converted_round_id = None
    _log(db, access, "round.delete", r.id, request, name=r.name)
    db.delete(r)
    db.commit()


class DealIn(BaseModel):
    investor_name: str = Field(min_length=1, max_length=120)
    firm: str | None = Field(default=None, max_length=120)
    stage: DealStage = DealStage.contacted
    amount: float | None = Field(default=None, ge=0)
    next_step: str | None = Field(default=None, max_length=200)
    next_date: date | None = None
    notes: str | None = Field(default=None, max_length=4000)


class DealUpdate(BaseModel):
    investor_name: str | None = Field(default=None, min_length=1, max_length=120)
    firm: str | None = Field(default=None, max_length=120)
    stage: DealStage | None = None
    amount: float | None = Field(default=None, ge=0)
    next_step: str | None = Field(default=None, max_length=200)
    next_date: date | None = None
    notes: str | None = Field(default=None, max_length=4000)


def _deal_out(d: Deal) -> dict:
    return {"id": str(d.id), "investor_name": d.investor_name, "firm": d.firm, "stage": str(d.stage),
            "amount": float(d.amount) if d.amount is not None else None, "next_step": d.next_step,
            "next_date": d.next_date.isoformat() if d.next_date else None, "notes": d.notes,
            "updated_at": d.updated_at.isoformat() if d.updated_at else None}


@router.get("/deals")
def list_deals(access: Access = Depends(startup_access()), db: Session = Depends(get_db)) -> list[dict]:
    return [_deal_out(d) for d in db.scalars(select(Deal).where(Deal.startup_id == access.startup.id).order_by(Deal.created_at))]


@router.post("/deals", status_code=status.HTTP_201_CREATED)
def add_deal(body: DealIn, request: Request, access: Access = Depends(startup_access()), db: Session = Depends(get_db)) -> dict:
    amount = None if body.amount is None else Decimal(str(body.amount))
    d = Deal(startup_id=access.startup.id, **{**body.model_dump(), "amount": amount})
    db.add(d)
    db.flush()
    _log(db, access, "deal.create", d.id, request, investor=d.investor_name)
    db.commit()
    return _deal_out(d)


@router.patch("/deals/{deal_id}")
def update_deal(deal_id: uuid.UUID, body: DealUpdate, request: Request, access: Access = Depends(startup_access()),
                db: Session = Depends(get_db)) -> dict:
    d = _own(db, Deal, deal_id, access)
    changes = body.model_dump(exclude_unset=True)
    for k, v in changes.items():
        setattr(d, k, Decimal(str(v)) if k == "amount" and v is not None else v)
    _log(db, access, "deal.update", d.id, request, **{k: str(v) for k, v in changes.items()})
    db.commit()
    return _deal_out(d)


@router.delete("/deals/{deal_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_deal(deal_id: uuid.UUID, request: Request, access: Access = Depends(startup_access()),
                db: Session = Depends(get_db)) -> None:
    d = _own(db, Deal, deal_id, access)
    _log(db, access, "deal.delete", d.id, request, investor=d.investor_name)
    db.delete(d)
    db.commit()
