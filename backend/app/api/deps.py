"""Who is calling, and may they touch this startup? Every startup-scoped endpoint depends on `startup_access`."""
import uuid
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.security import decode_access_token
from app.models import Membership, Role, Startup, User

_bearer = HTTPBearer(auto_error=False)

# Higher rank = more rights. v1 has founders only; roles added later slot in here and nowhere else.
ROLE_RANK: dict[Role, int] = {Role.founder: 100}


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer), db: Session = Depends(get_db)) -> User:
    unauthorized = HTTPException(status.HTTP_401_UNAUTHORIZED, "Not signed in", headers={"WWW-Authenticate": "Bearer"})
    if creds is None:
        raise unauthorized
    user_id = decode_access_token(creds.credentials)
    user = db.get(User, user_id) if user_id else None
    if user is None or not user.is_active:
        raise unauthorized
    return user


@dataclass
class Access:
    user: User
    startup: Startup
    role: Role


def startup_access(minimum: Role = Role.founder):
    """Dependency factory: the caller must be a member of the startup in the path with at least `minimum` rights.

    Non-members get 404, not 403, so the existence of other people's startups is never revealed."""

    def check(startup_id: uuid.UUID, user: User = Depends(current_user), db: Session = Depends(get_db)) -> Access:
        row = db.execute(
            select(Membership, Startup).join(Startup, Startup.id == Membership.startup_id)
            .where(Membership.user_id == user.id, Membership.startup_id == startup_id)
        ).first()
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Startup not found")
        membership, startup = row
        if ROLE_RANK[membership.role] < ROLE_RANK[minimum]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Your role does not allow this")
        return Access(user=user, startup=startup, role=membership.role)

    return check
