import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IdMixin, utcnow


class AuditLog(IdMixin, Base):
    """Append-only record of security-relevant and data-changing actions.

    A startup's rows form a hash chain (app/services/audit.py): each row stores the hash of the row before it, so an
    edited or removed row breaks the chain. Rows without a startup (sign-ins) are not chained.
    """

    __tablename__ = "audit_log"
    __table_args__ = (UniqueConstraint("startup_id", "seq", name="startup_seq"),)  # one chain per startup, never forked

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"), index=True)
    startup_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("startups.id", ondelete="CASCADE"), index=True)
    action: Mapped[str] = mapped_column(String(64))  # e.g. "auth.login", "startup.update"
    entity_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    ip: Mapped[str | None] = mapped_column(String(45))
    seq: Mapped[int | None] = mapped_column(Integer)          # position in the startup's chain, from 1
    prev_hash: Mapped[str | None] = mapped_column(String(64))
    hash: Mapped[str | None] = mapped_column(String(64))
