"""Cap table: who holds what, how it got there, and who the founder is talking to about the next round."""
import enum
import uuid
from datetime import date
from decimal import Decimal

from sqlalchemy import Date, Enum, ForeignKey, Integer, Numeric, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IdMixin, TimestampMixin


class HolderKind(enum.StrEnum):
    founder = "founder"
    investor = "investor"
    employee = "employee"
    advisor = "advisor"
    esop_pool = "esop_pool"  # options reserved for employees; grants are carved out of it
    other = "other"


class DealStage(enum.StrEnum):
    contacted = "contacted"
    meeting = "meeting"
    due_diligence = "due_diligence"
    term_sheet = "term_sheet"
    closed = "closed"
    passed = "passed"


def _enum(e: type[enum.Enum], name: str) -> Enum:
    return Enum(e, name=name, native_enum=False, length=16, validate_strings=True)


def _owner() -> Mapped[uuid.UUID]:
    return mapped_column(Uuid, ForeignKey("startups.id", ondelete="CASCADE"), index=True)


class Shareholder(IdMixin, TimestampMixin, Base):
    __tablename__ = "shareholders"

    startup_id: Mapped[uuid.UUID] = _owner()
    name: Mapped[str] = mapped_column(String(120))
    kind: Mapped[HolderKind] = mapped_column(_enum(HolderKind, "holder_kind"))
    email: Mapped[str | None] = mapped_column(String(320))


class FundingRound(IdMixin, TimestampMixin, Base):
    """A priced round. Shares bought in it are ShareIssue rows pointing here; SAFEs that converted point here too."""

    __tablename__ = "funding_rounds"

    startup_id: Mapped[uuid.UUID] = _owner()
    name: Mapped[str] = mapped_column(String(80))                       # "Pre-seed", "Seed"...
    closed_on: Mapped[date] = mapped_column(Date)
    pre_money: Mapped[Decimal] = mapped_column(Numeric(18, 2))
    amount_raised: Mapped[Decimal] = mapped_column(Numeric(18, 2))
    price_per_share: Mapped[Decimal] = mapped_column(Numeric(18, 6))


class ShareIssue(IdMixin, TimestampMixin, Base):
    """Shares given to a holder: at incorporation, in a round, on SAFE conversion, or into the ESOP pool."""

    __tablename__ = "share_issues"

    startup_id: Mapped[uuid.UUID] = _owner()
    shareholder_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("shareholders.id", ondelete="CASCADE"), index=True)
    round_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("funding_rounds.id", ondelete="SET NULL"))
    share_class: Mapped[str] = mapped_column(String(20), default="equity")   # equity | preference | pool
    shares: Mapped[int] = mapped_column(Integer)
    price_per_share: Mapped[Decimal | None] = mapped_column(Numeric(18, 6))
    issued_on: Mapped[date] = mapped_column(Date)


class Safe(IdMixin, TimestampMixin, Base):
    """A SAFE or convertible note: money in now, shares at the next priced round (cap and/or discount)."""

    __tablename__ = "safes"

    startup_id: Mapped[uuid.UUID] = _owner()
    shareholder_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("shareholders.id", ondelete="CASCADE"), index=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 2))
    valuation_cap: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))
    discount: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))         # 0.20 = 20%
    signed_on: Mapped[date] = mapped_column(Date)
    converted_round_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("funding_rounds.id", ondelete="SET NULL"))


class OptionGrant(IdMixin, TimestampMixin, Base):
    """Options granted to a person out of the ESOP pool, vesting monthly after a cliff."""

    __tablename__ = "option_grants"

    startup_id: Mapped[uuid.UUID] = _owner()
    shareholder_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("shareholders.id", ondelete="CASCADE"), index=True)
    shares: Mapped[int] = mapped_column(Integer)
    granted_on: Mapped[date] = mapped_column(Date)
    vesting_months: Mapped[int] = mapped_column(Integer, default=48)
    cliff_months: Mapped[int] = mapped_column(Integer, default=12)


class Deal(IdMixin, TimestampMixin, Base):
    """A conversation with an investor about the next round (the fundraising pipeline)."""

    __tablename__ = "deals"

    startup_id: Mapped[uuid.UUID] = _owner()
    investor_name: Mapped[str] = mapped_column(String(120))
    firm: Mapped[str | None] = mapped_column(String(120))
    stage: Mapped[DealStage] = mapped_column(_enum(DealStage, "deal_stage"), default=DealStage.contacted)
    amount: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))
    next_step: Mapped[str | None] = mapped_column(String(200))
    next_date: Mapped[date | None] = mapped_column(Date)
    notes: Mapped[str | None] = mapped_column(Text)
