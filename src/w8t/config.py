"""Runtime configuration for W8T.

Two run modes, selected by APP_ENV:
- "local": real personal data, persisted in SQLite. Never deployed publicly.
- "demo": synthetic/regeneratable data only, backed by Postgres (Neon) when
  a DATABASE_URL is provided. Used for the public Streamlit Cloud deployment.
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["local", "demo"] = "local"
    database_url: str = "sqlite:///./w8t_local.db"
    anthropic_api_key: str | None = None

    @property
    def is_demo(self) -> bool:
        return self.app_env == "demo"


settings = Settings()
