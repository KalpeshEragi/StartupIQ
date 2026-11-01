"""The audit log. A startup's rows form a hash chain: each row's hash covers its own content and the previous row's
hash, so editing or removing a row breaks every hash after it. Someone with database access could rebuild the whole
chain, which is why the latest hash (`head`) is printed on the verification report: a copy held by an investor
pins the history up to that point.
"""
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AuditLog, Startup
from app.models.base import utcnow


def _iso(d: datetime) -> str:
    # SQLite hands times back without a timezone; they are stored as UTC.
    return (d.replace(tzinfo=UTC) if d.tzinfo is None else d.astimezone(UTC)).isoformat(timespec="microseconds")


def digest(row: AuditLog) -> str:
    payload = {"seq": row.seq, "prev": row.prev_hash, "at": _iso(row.created_at), "action": row.action,
               "startup": str(row.startup_id), "user": str(row.user_id) if row.user_id else None,
               "entity": str(row.entity_id) if row.entity_id else None, "details": row.details, "ip": row.ip}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def record(db: Session, action: str, *, user_id: uuid.UUID | None = None, startup_id: uuid.UUID | None = None,
           entity_id: uuid.UUID | None = None, details: dict | None = None, request: Request | None = None) -> None:
    """Add an audit row to the current transaction: it is saved together with the change it describes, or not at all."""
    ip = request.client.host if request is not None and request.client else None
    # Round-trip through JSON so the hash covers exactly what the database will hand back.
    row = AuditLog(action=action, user_id=user_id, startup_id=startup_id, entity_id=entity_id, ip=ip, created_at=utcnow(),
                   details=json.loads(json.dumps(details or {}, default=str)))
    if startup_id is not None:
        db.flush()  # earlier rows of this request must be visible to the query below (sessions do not autoflush)
        # Lock the startup so two requests cannot both extend the chain from the same row (PostgreSQL; SQLite writes
        # one at a time anyway). The unique (startup_id, seq) constraint is the backstop.
        db.execute(select(Startup.id).where(Startup.id == startup_id).with_for_update())
        last = db.execute(select(AuditLog.seq, AuditLog.hash).where(AuditLog.startup_id == startup_id, AuditLog.seq.is_not(None))
                          .order_by(AuditLog.seq.desc()).limit(1)).first()
        row.seq, row.prev_hash = (last.seq + 1, last.hash) if last else (1, None)
        row.hash = digest(row)
    db.add(row)


@dataclass
class ChainCheck:
    ok: bool
    sealed: int            # rows in the chain
    unsealed: int          # older rows written before the chain existed
    head: str | None       # hash of the latest row
    problem: str | None = None


def verify_chain(db: Session, startup_id: uuid.UUID) -> ChainCheck:
    rows = db.scalars(select(AuditLog).where(AuditLog.startup_id == startup_id, AuditLog.seq.is_not(None))
                      .order_by(AuditLog.seq)).all()
    unsealed = len(db.scalars(select(AuditLog.id).where(AuditLog.startup_id == startup_id, AuditLog.seq.is_(None))).all())
    prev: str | None = None
    for expected, row in enumerate(rows, start=1):
        if row.seq != expected:
            return ChainCheck(False, len(rows), unsealed, rows[-1].hash, f"Entry {expected} of the activity history is missing.")
        if row.prev_hash != prev or row.hash != digest(row):
            return ChainCheck(False, len(rows), unsealed, rows[-1].hash,
                              f"Entry {expected} of the activity history ({row.action}) was changed after it was written.")
        prev = row.hash
    return ChainCheck(True, len(rows), unsealed, prev)
