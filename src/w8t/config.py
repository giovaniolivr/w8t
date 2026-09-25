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


# What reading the Streamlit secrets found - key NAMES only, never values - so the hosted app
# can explain a misconfiguration on screen (its logs are not visible from here).
SECRETS_STATUS: dict = {"read": False, "error": None, "keys": []}


def _streamlit_secrets() -> dict[str, str]:
    """Settings given as Streamlit secrets (hosted demo: Community Cloud's "Secrets" box).

    Read directly instead of relying on Streamlit copying root-level secrets into environment
    variables: that happens only once the secrets are parsed, and on the first hosted deploy the
    config had already been resolved without them (the app ran in local mode on an empty SQLite
    file). Keys are accepted at the root or inside one section (e.g. ``[general]``); root wins.
    Outside Streamlit, or with no secrets file, there is nothing to read.
    """
    try:
        import streamlit as st

        raw = st.secrets.to_dict()
    except Exception as exc:  # noqa: BLE001 - no runtime / no secrets file / parse error
        SECRETS_STATUS.update(read=False, error=type(exc).__name__, keys=[])
        return {}
    names = []
    for k, v in raw.items():
        names += [f"{k}.{sub}" for sub in v] if isinstance(v, dict) else [str(k)]
    SECRETS_STATUS.update(read=True, error=None, keys=sorted(names))

    fields = set(Settings.model_fields)
    found: dict[str, str] = {}
    for table in [raw, *(v for v in raw.values() if isinstance(v, dict))]:
        for k, v in table.items():
            key = str(k).lower()
            if key in fields and key not in found and isinstance(v, str | int | float):
                found[key] = str(v)
    return found


# Precedence: Streamlit secrets > environment variables > .env file > defaults.
settings = Settings(**_streamlit_secrets())
