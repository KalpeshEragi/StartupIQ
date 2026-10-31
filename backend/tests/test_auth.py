from sqlalchemy import select

from app.core.db import SessionLocal
from app.models import AuditLog, RefreshToken, User
from tests.conftest import PASSWORD, signup


def test_register_returns_token_and_sets_refresh_cookie(client):
    r = client.post("/api/v1/auth/register", json={"email": "Asha@Example.com", "full_name": " Asha ", "password": PASSWORD})
    assert r.status_code == 201
    body = r.json()
    assert body["user"]["email"] == "asha@example.com" and body["user"]["full_name"] == "Asha"
    assert "password" not in str(body) and body["expires_in"] == 15 * 60
    cookie = r.headers["set-cookie"].lower()
    assert "siq_refresh=" in cookie and "httponly" in cookie and "samesite=strict" in cookie and "path=/api/v1/auth" in cookie


def test_password_is_stored_hashed(client):
    signup(client)
    with SessionLocal() as db:
        user = db.scalar(select(User))
    assert user.password_hash.startswith("$argon2") and PASSWORD not in user.password_hash


def test_duplicate_email_is_rejected_case_insensitively(client):
    signup(client, "asha@example.com")
    r = client.post("/api/v1/auth/register", json={"email": "ASHA@example.com", "full_name": "Other", "password": PASSWORD})
    assert r.status_code == 409


def test_weak_or_malformed_input_is_rejected(client):
    for bad in ({"email": "a@b.co", "full_name": "A", "password": "short"},
                {"email": "nope", "full_name": "A", "password": PASSWORD}):
        assert client.post("/api/v1/auth/register", json=bad).status_code == 422


def test_login_and_me(client):
    signup(client)
    r = client.post("/api/v1/auth/login", json={"email": "founder@example.com", "password": PASSWORD})
    assert r.status_code == 200
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {r.json()['access_token']}"})
    assert me.status_code == 200 and me.json()["email"] == "founder@example.com"


def test_wrong_password_and_unknown_email_look_the_same(client):
    signup(client)
    a = client.post("/api/v1/auth/login", json={"email": "founder@example.com", "password": "wrong password!"})
    b = client.post("/api/v1/auth/login", json={"email": "ghost@example.com", "password": "wrong password!"})
    assert a.status_code == b.status_code == 401 and a.json() == b.json()
    with SessionLocal() as db:
        assert db.scalar(select(AuditLog).where(AuditLog.action == "auth.login_failed")) is not None


def test_protected_routes_need_a_valid_token(client):
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not-a-token"}).status_code == 401


def test_refresh_rotates_the_token(client):
    signup(client)
    old = client.cookies.get("siq_refresh")
    r = client.post("/api/v1/auth/refresh")
    assert r.status_code == 200 and r.json()["access_token"]
    assert client.cookies.get("siq_refresh") != old


def test_reusing_a_rotated_refresh_token_ends_the_session(client):
    signup(client)
    stolen = client.cookies.get("siq_refresh")
    assert client.post("/api/v1/auth/refresh").status_code == 200   # the real user refreshes
    client.cookies.set("siq_refresh", stolen, path="/api/v1/auth")
    assert client.post("/api/v1/auth/refresh").status_code == 401   # the attacker replays the old one
    with SessionLocal() as db:
        assert all(t.revoked_at is not None for t in db.scalars(select(RefreshToken)))  # every token in the family is dead


def test_logout_revokes_refresh(client):
    signup(client)
    token = client.cookies.get("siq_refresh")
    assert client.post("/api/v1/auth/logout").status_code == 204
    client.cookies.set("siq_refresh", token, path="/api/v1/auth")
    assert client.post("/api/v1/auth/refresh").status_code == 401
