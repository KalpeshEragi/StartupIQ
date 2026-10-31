import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IdMixin, utcnow


class AuditLog(IdMixin, Base):
    """Append-only record of security-relevant and data-changing actions."""

    __tablename__ = "audit_log"

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"), index=True)
    startup_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("startups.id", ondelete="CASCADE"), index=True)
    action: Mapped[str] = mapped_column(String(64))  # e.g. "auth.login", "startup.update"
    entity_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    ip: Mapped[str | None] = mapped_column(String(45))
