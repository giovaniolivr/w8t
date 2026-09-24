from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

import w8t.data.db as db_module
from w8t.config import settings
from w8t.data import repository
from w8t.data.models import Base
from w8t.insights import providers

TODAY = date.today()  # noqa: DTZ011 - entries are relative to "today" like real usage
PAGE_PATH = str(
    Path(__file__).resolve().parents[1] / "src" / "w8t" / "app" / "views" / "5_Resumo.py"
)


@pytest.fixture
def app(tmp_path, monkeypatch):
    test_engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(test_engine)
    monkeypatch.setattr(db_module, "engine", test_engine)
    monkeypatch.setattr(
        db_module, "SessionLocal", sessionmaker(bind=test_engine, expire_on_commit=False)
    )
    monkeypatch.setattr(settings, "app_env", "local")
    monkeypatch.setattr(settings, "gemini_api_key", None)
    return AppTest.from_file(PAGE_PATH, default_timeout=30)


def _seed(n=60):
    rng = np.random.default_rng(0)
    with db_module.get_session() as session:
        for i in range(n):
            repository.create_entry(
                session, entry_date=TODAY - timedelta(days=n - 1 - i),
                weight_kg=round(90 - 0.1 * i + rng.normal(0, 0.3), 1),
            )


DEFAULT_REPLY = "O peso atual segue a tendência."


class _FakeGemini(providers.FakeProvider):
    reply = DEFAULT_REPLY
    created = 0

    def __init__(self, api_key, model):
        super().__init__(type(self).reply)
        type(self).created += 1


@pytest.fixture
def fake_gemini(monkeypatch):
    _FakeGemini.created = 0
    _FakeGemini.reply = DEFAULT_REPLY
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(providers, "GeminiProvider", _FakeGemini)
    return _FakeGemini


def test_without_key_shows_numbers_and_how_to_configure(app):
    _seed()
    at = app.run()

    assert not at.exception
    assert "GEMINI_API_KEY" in at.info[0].value
    assert len(at.json) == 1  # the structured summary is still shown


def test_text_is_generated_only_on_click(app, fake_gemini):
    _seed()
    at = app.run()
    assert fake_gemini.created == 0  # no API call just by opening the page

    at.button[0].click().run()

    assert not at.exception
    assert fake_gemini.created == 1
    assert any("segue a tendência" in m.value for m in at.markdown)
    assert any("conferem com os números" in c.value for c in at.caption)


def test_invented_numbers_are_flagged(app, fake_gemini):
    _seed()
    fake_gemini.reply = "Em 12 semanas o peso chegará a 70,4 kg."
    at = app.run()
    at.button[0].click().run()

    assert not at.exception
    assert "12" in at.warning[0].value and "70,4" in at.warning[0].value


def test_demo_mode_shows_fixed_text_and_never_calls_the_api(app, fake_gemini, monkeypatch):
    _seed()
    monkeypatch.setattr(settings, "app_env", "demo")
    at = app.run()

    assert not at.exception
    assert fake_gemini.created == 0
    assert len(at.button) == 0
    assert any("Texto fixo" in c.value for c in at.caption)
