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
