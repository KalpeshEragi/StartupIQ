"""Tests run on a throwaway SQLite file by default; set SIQ_TEST_DATABASE_URL to run them on PostgreSQL (CI does)."""
import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="siq-tests-"))
os.environ["SIQ_DATABASE_URL"] = os.environ.get("SIQ_TEST_DATABASE_URL", f"sqlite:///{_TMP / 'test.db'}")
os.environ["SIQ_ENV"] = "test"
os.environ["SIQ_SECRET_KEY"] = "test-secret-key-that-is-long-enough"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Base  # noqa: E402

PASSWORD = "correct horse battery"
STARTUP = {"name": "Chai Labs", "sector": "D2C beverages", "stage": "seed", "product_type": "physical"}


@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def signup(client: TestClient, email: str = "founder@example.com", name: str = "Asha Founder") -> dict:
    r = client.post("/api/v1/auth/register", json={"email": email, "full_name": name, "password": PASSWORD})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}
