import enum
import uuid
from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Date, Enum, ForeignKey, SmallInteger, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, IdMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.user import User


class Stage(enum.StrEnum):
    idea = "idea"
    pre_seed = "pre_seed"
    seed = "seed"
    series_a = "series_a"
    series_b_plus = "series_b_plus"


class ProductType(enum.StrEnum):
    physical = "physical"
    digital = "digital"


class Role(enum.StrEnum):
    """v1 ships the founder role only; team and investor roles come in the roles phase (PRD section 11)."""

    founder = "founder"


def _enum(e: type[enum.Enum], name: str) -> Enum:
    return Enum(e, name=name, native_enum=False, length=32, validate_strings=True)


class Startup(IdMixin, TimestampMixin, Base):
    __tablename__ = "startups"
    __table_args__ = (CheckConstraint("fy_start_month BETWEEN 1 AND 12", name="fy_start_month"),)

    name: Mapped[str] = mapped_column(String(120))
    sector: Mapped[str] = mapped_column(String(80))
    stage: Mapped[Stage] = mapped_column(_enum(Stage, "stage"))
    product_type: Mapped[ProductType] = mapped_column(_enum(ProductType, "product_type"))
    founded_on: Mapped[date | None] = mapped_column(Date)
    currency: Mapped[str] = mapped_column(String(3), default="INR")
    fy_start_month: Mapped[int] = mapped_column(SmallInteger, default=4)  # April: the Indian financial year

    memberships: Mapped[list["Membership"]] = relationship(back_populates="startup", cascade="all, delete-orphan")


class Membership(IdMixin, TimestampMixin, Base):
    """Who may see a startup, and as what. Every startup-scoped endpoint checks this table (api/deps.py)."""

    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("user_id", "startup_id", name="user_startup"),)

    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    startup_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("startups.id", ondelete="CASCADE"), index=True)
    role: Mapped[Role] = mapped_column(_enum(Role, "role"))

    user: Mapped["User"] = relationship(back_populates="memberships")
    startup: Mapped[Startup] = relationship(back_populates="memberships")
