import pytest

from w8t.config import settings


@pytest.fixture(autouse=True)
def _local_mode_by_default(monkeypatch):
    """Tests never inherit the developer's run mode.

    ``settings`` is resolved from the real .env at import; with APP_ENV=demo there (e.g. while
    setting up the hosted demo) pages showed the demo banner/reset button and tests expecting the
    local mode failed. Tests that need demo mode still set it explicitly.
    """
    monkeypatch.setattr(settings, "app_env", "local")
