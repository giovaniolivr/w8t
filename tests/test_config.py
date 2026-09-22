from w8t.config import Settings


def test_defaults_to_local_sqlite():
    s = Settings(_env_file=None)
    assert s.app_env == "local"
    assert s.database_url.startswith("sqlite:///")
    assert s.is_demo is False
