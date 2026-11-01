"""Request and response shapes. Responses never include password hashes or token hashes."""
import uuid
from datetime import UTC, date, datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.models import ProductType, Role, Stage

# Stored times are UTC. SQLite hands them back without a timezone; mark them, so clients never read them as local time.
UtcDatetime = Annotated[datetime, AfterValidator(lambda d: d if d.tzinfo else d.replace(tzinfo=UTC))]


class RegisterIn(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=10, max_length=128)

    @field_validator("full_name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("full name is required")
        return v


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    full_name: str
    created_at: UtcDatetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105 (an OAuth token type, not a password)
    expires_in: int
    user: UserOut


class StartupIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    sector: str = Field(min_length=1, max_length=80)
    stage: Stage
    product_type: ProductType
    founded_on: date | None = None
    currency: str = Field(default="INR", pattern=r"^[A-Z]{3}$")
    fy_start_month: int = Field(default=4, ge=1, le=12)

    @field_validator("founded_on")
    @classmethod
    def _not_future(cls, v: date | None) -> date | None:
        if v is not None and v > date.today():
            raise ValueError("founding date cannot be in the future")
        return v


class StartupUpdate(BaseModel):
    """Every field optional: only the fields sent are changed."""

    name: str | None = Field(default=None, min_length=1, max_length=120)
    sector: str | None = Field(default=None, min_length=1, max_length=80)
    market_sector: str | None = None
    stage: Stage | None = None
    product_type: ProductType | None = None
    founded_on: date | None = None
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    fy_start_month: int | None = Field(default=None, ge=1, le=12)

    @field_validator("market_sector")
    @classmethod
    def _known_market(cls, v: str | None) -> str | None:
        from app.market.service import SECTORS
        if v is not None and v not in SECTORS:
            raise ValueError("unknown market sector")
        return v


class StartupOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    sector: str
    market_sector: str | None = None
    stage: Stage
    product_type: ProductType
    founded_on: date | None
    currency: str
    fy_start_month: int
    created_at: UtcDatetime
    role: Role | None = None  # the caller's role in this startup


class AuditOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    created_at: UtcDatetime
    action: str
    user_id: uuid.UUID | None
    details: dict


class BankAccountIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    bank_name: str | None = Field(default=None, max_length=80)
    last4: str | None = Field(default=None, pattern=r"^\d{4}$")
    currency: str = Field(default="INR", pattern=r"^[A-Z]{3}$")


class BankAccountOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    bank_name: str | None
    last4: str | None
    currency: str
    created_at: UtcDatetime


class CategoryOut(BaseModel):
    key: str
    label: str
    group: str  # "expense" (learned) or "structural" (from the bank format)


class TransactionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    bank_account_id: uuid.UUID
    txn_date: date
    description: str
    direction: str
    amount: float
    balance: float | None
    category: str | None
    category_source: str
    confidence: float | None
    needs_review: bool
    reason: str | None
    suggestions: list[str] = []
    source_row: int | None = None

    @field_validator("suggestions", mode="before")
    @classmethod
    def _split(cls, v: object) -> list[str]:
        return [s for s in v.split(",") if s] if isinstance(v, str) else (v or [])


class TransactionPage(BaseModel):
    items: list[TransactionOut]
    total: int


class ImportOut(BaseModel):
    id: uuid.UUID
    filename: str
    rows_read: int
    rows_added: int
    rows_duplicate: int
    rows_skipped: int
    period_start: date | None
    period_end: date | None
    columns: dict[str, str]
    notes: list[str]
    auto_filed: int        # added rows filed without review
    needs_review: int
    by_source: dict[str, int]
    file_sha256: str
    balance_status: str     # verified | warnings | unverified
    balance_checked: int
    balance_breaks: int
    issues: list[str]
    same_file_as: str | None = None  # an earlier upload of exactly this file


class CorrectionIn(BaseModel):
    category: str
    remember: bool = True  # also file this vendor's other and future transactions the same way


class CorrectionOut(BaseModel):
    transaction: TransactionOut
    also_updated: int


class ReviewSummary(BaseModel):
    total: int
    needs_review: int
    auto_filed_share: float
    by_source: dict[str, int]
    model_version: str | None


class ImportCheckOut(BaseModel):
    id: uuid.UUID
    filename: str
    account: str
    uploaded_at: UtcDatetime
    uploaded_by: str | None
    period_start: date | None
    period_end: date | None
    rows_added: int
    file_sha256: str | None
    status: str             # verified | warnings | unverified
    balance_checked: int
    balance_breaks: int
    opening_balance: float | None
    closing_balance: float | None
    issues: list[str]


class AuditChainOut(BaseModel):
    ok: bool
    sealed: int
    unsealed: int
    head: str | None
    problem: str | None


class TrustCounts(BaseModel):
    total: int
    verified: int       # from statements whose balances add up
    auto_filed: int     # filed by memory, rules or the model without review
    founder_set: int    # set or confirmed by the founder
    changed: int        # the founder's category differs from the app's first decision
    needs_review: int


class TrustMonth(TrustCounts):
    month: str          # YYYY-MM


class VerificationOut(BaseModel):
    generated_at: UtcDatetime
    status: str         # verified | warnings | unverified
    headline: str
    lines: TrustCounts
    months: list[TrustMonth]
    imports: list[ImportCheckOut]
    continuity: list[str]
    audit: AuditChainOut
    limits: str


class HistoryEvent(BaseModel):
    at: UtcDatetime
    what: str
    by: str | None = None


class TransactionHistory(BaseModel):
    transaction: TransactionOut
    file: str | None
    file_sha256: str | None
    statement_status: str | None
    source_row: int | None
    first_category: str | None
    first_source: str | None
    first_reason: str | None
    events: list[HistoryEvent]
