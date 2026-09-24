from datetime import date, timedelta
from pathlib import Path

import numpy as np
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
    assert _metric(at, "Ritmo (kg/sem)").value == "-0.70"
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


def test_home_patterns_section(app):
    # Steady loss with one planted spike -> trend down, no plateau, the spike flagged.
    start = TODAY - timedelta(days=39)
    with db_module.get_session() as session:
        for i in range(40):
            weight = 90.0 - 0.1 * i + (0.1 if i % 2 else -0.1) + (3.0 if i == 30 else 0.0)
            repository.create_entry(
                session, entry_date=start + timedelta(days=i), weight_kg=round(weight, 1)
            )

    at = app.run()

    assert not at.exception
    # 40 days of data: enough for the Kalman detectors (the dashboard says which one ran).
    assert _metric(at, "Tendência (Kalman, 60 dias)").value == "descendo"
    assert _metric(at, "Platôs detectados").value == "0"
    assert _metric(at, "Medições atípicas").value == "1"
    anomalies_table = at.dataframe[-1].value
    assert anomalies_table["Data"].iloc[0] == (start + timedelta(days=30)).strftime("%d/%m/%Y")


def test_short_history_falls_back_to_baseline_detectors(app):
    # 10 daily points: too short for the Kalman model (needs 14 over 21 days).
    start = TODAY - timedelta(days=9)
    with db_module.get_session() as session:
        for i in range(10):
            repository.create_entry(
                session, entry_date=start + timedelta(days=i), weight_kg=90.0 - 0.1 * i
            )

    at = app.run()

    assert not at.exception
    assert _metric(at, "Tendência (21 dias)").value == "descendo"
    assert not any("Kalman" in c.value for c in at.caption)


def test_weekly_pattern_is_announced(app):
    rng = np.random.default_rng(1)
    start = TODAY - timedelta(days=119)
    with db_module.get_session() as session:
        for i in range(120):
            weekend = 0.6 if (start + timedelta(days=i)).weekday() >= 5 else 0.0
            repository.create_entry(
                session, entry_date=start + timedelta(days=i),
                weight_kg=round(80 + weekend + rng.normal(0, 0.25), 1),
            )

    at = app.run()

    assert not at.exception
    assert any("Padrão semanal detectado" in c.value for c in at.caption)


def _seed_reversal(goal):
    """70 days losing, then 40 days clearly gaining, inside one open-ended period."""
    rng = np.random.default_rng(3)
    start = TODAY - timedelta(days=109)
    with db_module.get_session() as session:
        for i in range(110):
            w = 90 - 0.1 * i if i < 70 else 83 + 0.12 * (i - 70)
            repository.create_entry(
                session, entry_date=start + timedelta(days=i),
                weight_kg=round(w + rng.normal(0, 0.2), 1),
            )
        periods.create_period(session, label="Fase", goal_direction=goal, start_date=start)


def test_dashboard_opens_on_the_current_period(app):
    _seed_reversal(GoalDirection.GAIN)
    at = app.run()

    assert not at.exception
    assert at.selectbox[0].value.startswith("Fase")


def test_reversal_against_goal_suggests_a_new_period(app):
    _seed_reversal(GoalDirection.LOSS)
    at = app.run()

    assert not at.exception
    assert any("O peso começou a subir" in m.value for m in at.markdown)
    assert any(b.label == "Iniciar novo período" for b in at.button)

    next(b for b in at.button if b.label == "Ignorar").click().run()
    assert not any("O peso começou a subir" in m.value for m in at.markdown)


def test_no_suggestion_when_trend_matches_goal(app):
    _seed_reversal(GoalDirection.GAIN)
    at = app.run()

    assert not any("O peso começou" in m.value for m in at.markdown)


def test_dashboard_opens_the_period_requested_in_the_url(app):
    _seed_reversal(GoalDirection.GAIN)
    with db_module.get_session() as session:
        old = periods.create_period(
            session, label="Antigo", goal_direction=GoalDirection.LOSS,
            start_date=TODAY - timedelta(days=200), end_date=TODAY - timedelta(days=150),
        )
        old_id = old.id

    app.query_params["periodo"] = str(old_id)
    at = app.run()

    assert not at.exception
    assert at.selectbox[0].value.startswith("Antigo")
