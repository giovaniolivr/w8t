"""Runtime configuration for W8T.

Two run modes, selected by APP_ENV:
- "local": real personal data, persisted in SQLite. Never deployed publicly.
- "demo": synthetic/regeneratable data only, backed by Postgres (Neon) when
  a DATABASE_URL is provided. Used for the public Streamlit Cloud deployment.
"""

from __future__ import annotations

from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["local", "demo"] = "local"
    database_url: str = "sqlite:///./w8t_local.db"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash"

    @field_validator("database_url")
    @classmethod
    def _use_psycopg3(cls, url: str) -> str:
        """Neon (and most hosts) hand out ``postgresql://`` / ``postgres://`` URLs, which
        SQLAlchemy maps to psycopg2 - not installed. Point them at psycopg 3."""
        for prefix in ("postgresql://", "postgres://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url[len(prefix):]
        return url

    @property
    def is_demo(self) -> bool:
        return self.app_env == "demo"


settings = Settings()
