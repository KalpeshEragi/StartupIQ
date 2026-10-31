"""Sign up, log in, refresh, log out.

The access token (15 min) goes in the response body and the browser keeps it in memory. The refresh token (30 days)
goes in an httpOnly cookie scoped to /api/v1/auth, so page scripts can never read it.
"""
import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import current_user
from app.core.config import get_settings
from app.core.db import get_db
from app.core.security import (
    create_access_token,
    hash_password,
    hash_refresh_token,
    needs_rehash,
    new_refresh_token,
    verify_password,
)
from app.models import RefreshToken, User
from app.schemas import LoginIn, RegisterIn, TokenOut, UserOut
from app.services import audit

router = APIRouter(prefix="/auth", tags=["auth"])
COOKIE = "siq_refresh"
COOKIE_PATH = "/api/v1/auth"


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)  # SQLite returns naive datetimes


def _issue(db: Session, response: Response, user: User, family_id: uuid.UUID | None = None) -> TokenOut:
    s = get_settings()
    token, token_hash = new_refresh_token()
    db.add(RefreshToken(user_id=user.id, family_id=family_id or uuid.uuid4(), token_hash=token_hash,
                        expires_at=datetime.now(UTC) + timedelta(days=s.refresh_token_days)))
    response.set_cookie(COOKIE, token, max_age=s.refresh_token_days * 86400, httponly=True, secure=s.cookie_secure,
                        samesite="strict", path=COOKIE_PATH)
    access, expires_in = create_access_token(user.id)
    return TokenOut(access_token=access, expires_in=expires_in, user=UserOut.model_validate(user))


@router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
def register(body: RegisterIn, request: Request, response: Response, db: Session = Depends(get_db)) -> TokenOut:
    user = User(email=body.email.lower(), full_name=body.full_name, password_hash=hash_password(body.password),
                last_login_at=datetime.now(UTC))
    db.add(user)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists") from None
    audit.record(db, "auth.register", user_id=user.id, request=request)
    out = _issue(db, response, user)
    db.commit()
    return out


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, request: Request, response: Response, db: Session = Depends(get_db)) -> TokenOut:
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if not verify_password(body.password, user.password_hash if user else None) or not user.is_active:
        audit.record(db, "auth.login_failed", user_id=user.id if user else None, details={"email": body.email.lower()},
                     request=request)
        db.commit()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(body.password)
    user.last_login_at = datetime.now(UTC)
    audit.record(db, "auth.login", user_id=user.id, request=request)
    out = _issue(db, response, user)
    db.commit()
    return out


@router.post("/refresh", response_model=TokenOut)
def refresh(request: Request, response: Response, db: Session = Depends(get_db),
            token: str | None = Cookie(default=None, alias=COOKIE)) -> TokenOut:
    expired = HTTPException(status.HTTP_401_UNAUTHORIZED, "Session expired, please sign in again")
    row = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(token))) if token else None
    if row is None:
        raise expired
    now = datetime.now(UTC)
    if row.revoked_at is not None:  # a rotated token came back: treat as stolen and end the whole session family
        db.execute(update(RefreshToken).where(RefreshToken.family_id == row.family_id, RefreshToken.revoked_at.is_(None))
                   .values(revoked_at=now))
        audit.record(db, "auth.refresh_reuse", user_id=row.user_id, request=request)
        db.commit()
        response.delete_cookie(COOKIE, path=COOKIE_PATH)
        raise expired
    user = db.get(User, row.user_id)
    if _as_utc(row.expires_at) <= now or user is None or not user.is_active:
        raise expired
    row.revoked_at = now
    out = _issue(db, response, user, family_id=row.family_id)
    db.commit()
    return out


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, response: Response, db: Session = Depends(get_db),
           token: str | None = Cookie(default=None, alias=COOKIE)) -> None:
    row = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(token))) if token else None
    if row is not None:
        db.execute(update(RefreshToken).where(RefreshToken.family_id == row.family_id, RefreshToken.revoked_at.is_(None))
                   .values(revoked_at=datetime.now(UTC)))
        audit.record(db, "auth.logout", user_id=row.user_id, request=request)
        db.commit()
    response.delete_cookie(COOKIE, path=COOKIE_PATH)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)) -> User:
    return user
