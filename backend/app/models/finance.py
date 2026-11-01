"""Bank accounts, imported statements, transactions, and what the founder has taught the categoriser."""
import enum
import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IdMixin, TimestampMixin, utcnow


class Direction(enum.StrEnum):
    debit = "debit"    # money out
    credit = "credit"  # money in


class CategorySource(enum.StrEnum):
    """How a transaction got its category, shown to the founder next to it."""

    founder = "founder"  # set or corrected by hand
    memory = "memory"    # the founder's earlier correction for this vendor
    rule = "rule"        # recognised from the bank narration format
    model = "model"      # predicted by the learned categoriser
    none = "none"        # nothing confident enough: waiting for review


def _enum(e: type[enum.Enum], name: str) -> Enum:
    return Enum(e, name=name, native_enum=False, length=16, validate_strings=True)


class BankAccount(IdMixin, TimestampMixin, Base):
    __tablename__ = "bank_accounts"

    startup_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("startups.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(80))                   # e.g. "HDFC current account"
    bank_name: Mapped[str | None] = mapped_column(String(80))
    last4: Mapped[str | None] = mapped_column(String(4))            # never the full account number
    currency: Mapped[str] = mapped_column(String(3), default="INR")
    # Balance before the first imported line; only needed when the bank's export has no balance column.
    opening_balance: Mapped[Decimal | None] = mapped_column(Numeric(16, 2))


class StatementImport(IdMixin, Base):
    """One uploaded statement file and what happened to its rows."""

    __tablename__ = "statement_imports"

    startup_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("startups.id", ondelete="CASCADE"), index=True)
    bank_account_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("bank_accounts.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    rows_read: Mapped[int] = mapped_column(Integer, default=0)
    rows_added: Mapped[int] = mapped_column(Integer, default=0)
    rows_duplicate: Mapped[int] = mapped_column(Integer, default=0)   # already imported from an earlier file
    rows_skipped: Mapped[int] = mapped_column(Integer, default=0)     # unreadable date or amount
    period_start: Mapped[date | None] = mapped_column(Date)
    period_end: Mapped[date | None] = mapped_column(Date)
    # Verification (app/services/verification.py): the file's fingerprint and whether its balance column adds up.
    file_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    rows_sha256: Mapped[str | None] = mapped_column(String(64))       # seal over the lines this import added
    balance_status: Mapped[str | None] = mapped_column(String(16))   # verified | warnings | unverified
    balance_checked: Mapped[int] = mapped_column(Integer, default=0)  # consecutive line pairs checked
    balance_breaks: Mapped[int] = mapped_column(Integer, default=0)   # pairs where the balance does not add up
    opening_balance: Mapped[Decimal | None] = mapped_column(Numeric(16, 2))  # before the earliest line
    closing_balance: Mapped[Decimal | None] = mapped_column(Numeric(16, 2))  # after the latest line
    issues: Mapped[list] = mapped_column(JSON, default=list)          # plain-language findings shown to the founder


class Transaction(IdMixin, TimestampMixin, Base):
    __tablename__ = "transactions"
    __table_args__ = (
        UniqueConstraint("bank_account_id", "fingerprint", name="account_fingerprint"),  # the same line is never imported twice
        Index("ix_transactions_startup_date", "startup_id", "txn_date"),
    )

    startup_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("startups.id", ondelete="CASCADE"))
    bank_account_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("bank_accounts.id", ondelete="CASCADE"), index=True)
    import_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("statement_imports.id", ondelete="SET NULL"))
    source_row: Mapped[int | None] = mapped_column(Integer)  # line number in the uploaded file
    txn_date: Mapped[date] = mapped_column(Date)
    description: Mapped[str] = mapped_column(String(500))
    direction: Mapped[Direction] = mapped_column(_enum(Direction, "direction"))
    amount: Mapped[Decimal] = mapped_column(Numeric(14, 2))         # always positive; direction says in or out
    balance: Mapped[Decimal | None] = mapped_column(Numeric(16, 2))
    fingerprint: Mapped[str] = mapped_column(String(64))
    vendor_key: Mapped[str] = mapped_column(String(200), index=True)  # cleaned description, used by the founder's memory

    category: Mapped[str | None] = mapped_column(String(40), index=True)
    category_source: Mapped[CategorySource] = mapped_column(_enum(CategorySource, "category_source"), default=CategorySource.none)
    confidence: Mapped[float | None] = mapped_column(Float)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    reason: Mapped[str | None] = mapped_column(String(300))
    suggestions: Mapped[str | None] = mapped_column(String(160))  # comma-separated category keys, best first, when unsure
    model_version: Mapped[str | None] = mapped_column(String(40))
    # The app's first decision, kept as it was: corrections change category/source/reason above, never these.
    auto_category: Mapped[str | None] = mapped_column(String(40))
    auto_source: Mapped[str | None] = mapped_column(String(16))
    auto_reason: Mapped[str | None] = mapped_column(String(300))


class VendorRule(IdMixin, TimestampMixin, Base):
    """The founder's memory: lines whose cleaned description matches `vendor_key` get `category` straight away."""

    __tablename__ = "vendor_rules"
    __table_args__ = (UniqueConstraint("startup_id", "vendor_key", name="startup_vendor"),)

    startup_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("startups.id", ondelete="CASCADE"), index=True)
    vendor_key: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(40))
    created_by: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"))
    times_applied: Mapped[int] = mapped_column(Integer, default=0)


class CategoryCorrection(IdMixin, Base):
    """Every change a founder makes: the labelled examples the next model is trained on (with the startup's consent)."""

    __tablename__ = "category_corrections"

    startup_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("startups.id", ondelete="CASCADE"), index=True)
    transaction_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("transactions.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    created_by: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"))
    previous_category: Mapped[str | None] = mapped_column(String(40))
    previous_source: Mapped[str | None] = mapped_column(String(16))
    new_category: Mapped[str] = mapped_column(String(40))
    text: Mapped[str] = mapped_column(String(200))   # cleaned description at the time
    direction: Mapped[str] = mapped_column(String(8))
    amount: Mapped[Decimal] = mapped_column(Numeric(14, 2))


class UnitMetric(IdMixin, TimestampMixin, Base):
    """The few monthly numbers a bank statement cannot know. Every field is optional; ratios are computed (finance/)."""

    __tablename__ = "unit_metrics"
    __table_args__ = (UniqueConstraint("startup_id", "month", name="startup_month"),)

    startup_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("startups.id", ondelete="CASCADE"), index=True)
    month: Mapped[date] = mapped_column(Date)                         # first day of the month
    # physical products
    units_sold: Mapped[int | None] = mapped_column(Integer)
    orders: Mapped[int | None] = mapped_column(Integer)
    inventory_value: Mapped[Decimal | None] = mapped_column(Numeric(16, 2))
    # digital products
    customers: Mapped[int | None] = mapped_column(Integer)              # paying customers at month end
    new_customers: Mapped[int | None] = mapped_column(Integer)
    churned_customers: Mapped[int | None] = mapped_column(Integer)
    mrr: Mapped[Decimal | None] = mapped_column(Numeric(16, 2))         # monthly recurring revenue at month end
