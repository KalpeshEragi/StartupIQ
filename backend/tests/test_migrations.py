"""The Alembic migrations must build exactly the schema the models describe."""
import tempfile
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine

from app.core.config import get_settings
from app.models import Base

ROOT = Path(__file__).resolve().parents[1]


def test_migrations_match_models(monkeypatch):
    url = f"sqlite:///{Path(tempfile.mkdtemp()) / 'migrate.db'}"
    monkeypatch.setenv("SIQ_DATABASE_URL", url)
    get_settings.cache_clear()
    try:
        cfg = Config(str(ROOT / "alembic.ini"))
        command.upgrade(cfg, "head")
        engine = create_engine(url)
        with engine.connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
        engine.dispose()
        assert diff == [], diff
        command.downgrade(cfg, "base")
    finally:
        get_settings.cache_clear()
