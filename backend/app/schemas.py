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
    stage: Stage | None = None
    product_type: ProductType | None = None
    founded_on: date | None = None
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    fy_start_month: int | None = Field(default=None, ge=1, le=12)


class StartupOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    sector: str
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
