from datetime import date, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

import w8t.data.db as db_module
from w8t.data import periods, repository
from w8t.data.models import Base, GoalDirection

TODAY = date.today()  # noqa: DTZ011 - Home.py scopes open periods with date.today()
PAGE_PATH = str(Path(__file__).resolve().parents[1] / "src" / "w8t" / "app" / "Home.py")


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

    return AppTest.from_file(PAGE_PATH, default_timeout=15)


def _metric(at, label):
    return next(m for m in at.metric if m.label == label)


def test_home_with_no_entries_shows_empty_state(app):
    at = app.run()

    assert not at.exception
    assert "Nenhum registro de peso" in at.info[0].value


def test_home_shows_kpis_for_full_history(app):
    # 5 measurements every other day: enough for pace and the 7-day mean, not the 30-day one.
    start = TODAY - timedelta(days=8)
    with db_module.get_session() as session:
        for i in range(5):
            repository.create_entry(
                session, entry_date=start + timedelta(days=2 * i), weight_kg=90.0 - 0.2 * i
            )

    at = app.run()

    assert not at.exception
    assert _metric(at, "Peso atual").value == "89.2 kg"
    assert _metric(at, "Peso inicial").value == "90.0 kg"
    assert _metric(at, "Ritmo").value == "-0.70 kg/sem"
    assert _metric(at, "Média móvel 30 dias").value == "dados insuficientes"


def test_home_scoped_to_period(app):
    start = TODAY - timedelta(days=9)
    with db_module.get_session() as session:
        for i in range(10):
            repository.create_entry(
                session, entry_date=start + timedelta(days=i), weight_kg=80.0 + i
            )
        periods.create_period(
            session,
            label="Bulk",
            goal_direction=GoalDirection.GAIN,
            start_date=start + timedelta(days=5),
            target_weight_kg=95.0,
        )

    at = app.run()
    at.selectbox[0].select_index(1).run()

    assert not at.exception
    assert _metric(at, "Peso inicial").value == "85.0 kg"
    assert _metric(at, "Peso atual").value == "89.0 kg"
    assert any("Meta do período" in c.value for c in at.caption)


def test_home_period_without_entries(app):
    with db_module.get_session() as session:
        repository.create_entry(
            session, entry_date=TODAY - timedelta(days=30), weight_kg=80.0
        )
        periods.create_period(
            session,
            label="Cut",
            goal_direction=GoalDirection.LOSS,
            start_date=TODAY - timedelta(days=5),
        )

    at = app.run()
    at.selectbox[0].select_index(1).run()

    assert not at.exception
    assert "Nenhum registro de peso dentro deste período" in at.info[0].value


def test_demo_reset_button_only_in_demo_mode(app, monkeypatch):
    from w8t.config import settings

    at = app.run()
    assert len(at.sidebar.button) == 0

    monkeypatch.setattr(settings, "app_env", "demo")
    at = app.run()
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert "Modo demonstração" in at.warning[0].value
    assert _metric(at, "Peso atual").value.endswith("kg")
    assert len(at.selectbox[0].options) == 3  # histórico completo + 2 períodos sintéticos
