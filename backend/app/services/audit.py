import uuid

from fastapi import Request
from sqlalchemy.orm import Session

from app.models import AuditLog


def record(db: Session, action: str, *, user_id: uuid.UUID | None = None, startup_id: uuid.UUID | None = None,
           entity_id: uuid.UUID | None = None, details: dict | None = None, request: Request | None = None) -> None:
    """Add an audit row to the current transaction: it is saved together with the change it describes, or not at all."""
    ip = request.client.host if request is not None and request.client else None
    db.add(AuditLog(action=action, user_id=user_id, startup_id=startup_id, entity_id=entity_id, details=details or {}, ip=ip))
