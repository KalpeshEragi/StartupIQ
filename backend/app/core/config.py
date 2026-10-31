"""Settings from the environment (or a .env file). Every value has a safe local default; production must set a real secret."""
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SIQ_", extra="ignore")

    env: str = "local"  # local | test | production
    database_url: str = "postgresql+psycopg://startupiq:startupiq@localhost:5432/startupiq"
    secret_key: str = Field(default="dev-only-change-me-please", min_length=16)
    access_token_minutes: int = 15
    refresh_token_days: int = 30
    cors_origins: list[str] = ["http://localhost:5173"]
    cookie_secure: bool = False  # True behind HTTPS

    @model_validator(mode="after")
    def _production_needs_a_real_secret(self):
        if self.env == "production" and (self.secret_key.startswith("dev-") or len(self.secret_key) < 32):
            raise ValueError("SIQ_SECRET_KEY must be a random value of 32+ characters in production")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
