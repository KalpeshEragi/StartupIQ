from fastapi import APIRouter, Depends, Query, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import Access, current_user, startup_access
from app.core.db import get_db
from app.models import AuditLog, Membership, Role, Startup, User
from app.schemas import AuditOut, StartupIn, StartupOut, StartupUpdate
from app.services import audit

router = APIRouter(prefix="/startups", tags=["startups"])


def _out(startup: Startup, role: Role) -> StartupOut:
    return StartupOut.model_validate(startup).model_copy(update={"role": role})


@router.get("", response_model=list[StartupOut])
def list_startups(user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[StartupOut]:
    rows = db.execute(select(Startup, Membership.role).join(Membership).where(Membership.user_id == user.id)
                      .order_by(Startup.created_at)).all()
    return [_out(s, role) for s, role in rows]


@router.post("", response_model=StartupOut, status_code=status.HTTP_201_CREATED)
def create_startup(body: StartupIn, request: Request, user: User = Depends(current_user),
                   db: Session = Depends(get_db)) -> StartupOut:
    startup = Startup(**body.model_dump())
    db.add(startup)
    db.flush()
    db.add(Membership(user_id=user.id, startup_id=startup.id, role=Role.founder))
    audit.record(db, "startup.create", user_id=user.id, startup_id=startup.id, entity_id=startup.id,
                 details={"name": startup.name}, request=request)
    db.commit()
    return _out(startup, Role.founder)


@router.get("/{startup_id}", response_model=StartupOut)
def get_startup(access: Access = Depends(startup_access())) -> StartupOut:
    return _out(access.startup, access.role)


@router.patch("/{startup_id}", response_model=StartupOut)
def update_startup(body: StartupUpdate, request: Request, access: Access = Depends(startup_access(Role.founder)),
                   db: Session = Depends(get_db)) -> StartupOut:
    changes = body.model_dump(exclude_unset=True)
    startup = access.startup
    before = {k: getattr(startup, k) for k in changes}
    for key, value in changes.items():
        setattr(startup, key, value)
    changed = {k: {"from": str(before[k]), "to": str(v)} for k, v in changes.items() if before[k] != v}
    if changed:
        audit.record(db, "startup.update", user_id=access.user.id, startup_id=startup.id, entity_id=startup.id,
                     details=changed, request=request)
    db.commit()
    return _out(startup, access.role)


@router.get("/{startup_id}/activity", response_model=list[AuditOut])
def activity(access: Access = Depends(startup_access(Role.founder)), db: Session = Depends(get_db),
             limit: int = Query(default=50, ge=1, le=200)) -> list[AuditLog]:
    return list(db.scalars(select(AuditLog).where(AuditLog.startup_id == access.startup.id)
                           .order_by(AuditLog.created_at.desc()).limit(limit)))
