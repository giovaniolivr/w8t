from w8t.config import Settings


def test_defaults_to_local_sqlite():
    s = Settings(_env_file=None)
    assert s.app_env == "local"
    assert s.database_url.startswith("sqlite:///")
    assert s.is_demo is False


def test_hosted_postgres_urls_use_psycopg3():
    host = "user:pw@ep-x.neon.tech/neondb?sslmode=require&channel_binding=require"
    for scheme in ("postgresql://", "postgres://"):
        s = Settings(_env_file=None, database_url=scheme + host)
        assert s.database_url == "postgresql+psycopg://" + host
    explicit = "postgresql+psycopg://" + host
    assert Settings(_env_file=None, database_url=explicit).database_url == explicit


def test_streamlit_secrets_are_read_directly(monkeypatch):
    import streamlit as st

    from w8t import config

    fake = {"APP_ENV": "demo", "DATABASE_URL": "postgres://u:p@h/db", "OTHER": "x",
            "section": {"nested": "ignored"}}
    monkeypatch.setattr(st, "secrets", type("S", (), {"to_dict": lambda self: fake})())
    values = config._streamlit_secrets()
    assert values == {"app_env": "demo", "database_url": "postgres://u:p@h/db"}
    s = config.Settings(_env_file=None, **values)
    assert s.is_demo and s.database_url.startswith("postgresql+psycopg://")


def test_no_secrets_file_means_no_overrides():
    from w8t import config

    assert config._streamlit_secrets() == {}  # tests run without .streamlit/secrets.toml
