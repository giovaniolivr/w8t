from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

import w8t.data.db as db_module
from w8t.data.models import Base

PAGE_PATH = str(
    Path(__file__).resolve().parents[1] / "src" / "w8t" / "app" / "views" / "2_Periodos.py"
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

    return AppTest.from_file(PAGE_PATH, default_timeout=15)


def test_page_loads_with_no_periods(app):
    at = app.run()

    assert not at.exception
    assert "Nenhum período cadastrado" in at.info[0].value


def test_creating_a_period_shows_up_in_history(app):
    at = app.run()

    at.text_input[0].set_value("Cutting verão")
    at.get_by_key("FormSubmitter:new_period_form-Salvar").click().run()

    assert not at.exception
    assert at.success[0].value.startswith("Período 'Cutting verão'")
    assert "Cutting verão" in at.dataframe[0].value.to_string()


def test_overlapping_period_is_rejected_in_ui(app):
    at = app.run()

    at.text_input[0].set_value("Bulk")
    at.date_input[0].set_value(at.date_input[0].value)
    at.checkbox[0].set_value(True)  # "em andamento" -> sem data de fim
    at.get_by_key("FormSubmitter:new_period_form-Salvar").click().run()

    at.text_input[0].set_value("Cutting")
    at.get_by_key("FormSubmitter:new_period_form-Salvar").click().run()

    assert not at.exception
    assert "Sobrepõe o período existente" in at.error[0].value


def test_loose_entries_can_be_attached_to_a_period(app):
    from datetime import date, timedelta

    from w8t.data import periods, repository
    from w8t.data.models import GoalDirection

    today = date.today()  # noqa: DTZ011
    with db_module.get_session() as session:
        for i in range(7, 14):  # a week logged before any period existed
            repository.create_entry(session, entry_date=today - timedelta(days=i), weight_kg=80.0)
        periods.create_period(
            session, label="Cut", goal_direction=GoalDirection.LOSS,
            start_date=today - timedelta(days=6),
        )

    at = app.run()
    assert any("7 registro(s) sem período" in m.value for m in at.markdown)

    next(b for b in at.button if b.label.startswith("Vincular 7 registro(s) a 'Cut'")).click().run()

    assert not at.exception
    assert "7 registro(s) vinculados" in at.success[0].value
    with db_module.get_session() as session:
        assert periods.uncovered_entries(session) == []


def test_prefill_for_loose_entries_sets_start_date(app):
    from datetime import date, timedelta

    from w8t.data import repository

    first = date.today() - timedelta(days=20)  # noqa: DTZ011
    with db_module.get_session() as session:
        repository.create_entry(session, entry_date=first, weight_kg=80.0)

    at = app.run()
    next(b for b in at.button if b.label == "Criar período para esses registros").click().run()

    assert not at.exception
    assert at.date_input(key="new_period_start").value == first
