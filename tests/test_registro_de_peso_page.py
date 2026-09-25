from datetime import date, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

import w8t.data.db as db_module
from w8t.data import periods, repository
from w8t.data.models import Base, GoalDirection

TODAY = date.today()  # noqa: DTZ011 - the page defaults to the local calendar day
PAGE_PATH = str(
    Path(__file__).resolve().parents[1] / "src" / "w8t" / "app" / "views" / "1_Registro_de_Peso.py"
)


@pytest.fixture
def app(tmp_path, monkeypatch):
    # w8t.config.settings is a singleton resolved at first import, so it can already be
    # bound elsewhere by the time this fixture runs (e.g. another test file imported it
    # first) - patching env vars here would be too late. Swap the db module's engine and
    # session factory directly instead, which get_session() always looks up at call time.
    test_engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(test_engine)
    monkeypatch.setattr(db_module, "engine", test_engine)
    monkeypatch.setattr(
        db_module, "SessionLocal", sessionmaker(bind=test_engine, expire_on_commit=False)
    )

    return AppTest.from_file(PAGE_PATH, default_timeout=15)


def test_page_loads_with_empty_history(app):
    at = app.run()

    assert not at.exception
    assert "Nenhum registro ainda" in at.info[0].value


def _open_period(start):
    with db_module.get_session() as session:
        periods.create_period(
            session, label="Cut", goal_direction=GoalDirection.LOSS, start_date=start
        )


def _submit(at, weight):
    at.number_input[0].set_value(weight)
    return at.get_by_key("FormSubmitter:new_entry_form-Salvar").click().run()


def test_entry_inside_a_period_is_saved_directly(app):
    _open_period(TODAY - timedelta(days=30))
    at = _submit(app.run(), 80.5)

    assert not at.exception
    assert at.success[0].value.startswith("Registro de")
    assert "Cut" in at.dataframe[0].value.to_string()


def test_entry_without_period_asks_for_goal_before_saving(app):
    at = _submit(app.run(), 80.5)

    assert not at.exception
    assert any("Qual é o objetivo" in m.value for m in at.markdown)
    with db_module.get_session() as session:
        assert repository.list_entries(session) == []  # nothing saved yet

    at.radio[0].set_value(GoalDirection.GAIN).run()
    next(b for b in at.button if b.label == "Criar período e salvar registro").click().run()

    assert not at.exception
    assert at.success[0].value.startswith("Registro de")
    with db_module.get_session() as session:
        [p] = periods.list_periods(session)
        assert p.goal_direction is GoalDirection.GAIN and p.end_date is None
        assert len(repository.list_entries(session)) == 1


def test_new_period_absorbs_earlier_entries_without_period(app):
    today = TODAY
    with db_module.get_session() as session:
        for i in range(3, 6):
            repository.create_entry(session, entry_date=today - timedelta(days=i), weight_kg=81.0)

    at = _submit(app.run(), 80.5)
    assert any("incluirá 3 registro(s)" in c.value for c in at.caption)
    next(b for b in at.button if b.label == "Criar período e salvar registro").click().run()

    assert not at.exception
    with db_module.get_session() as session:
        assert periods.uncovered_entries(session) == []
        assert periods.list_periods(session)[0].start_date == today - timedelta(days=5)


def test_cancel_discards_the_pending_entry(app):
    at = _submit(app.run(), 80.5)
    next(b for b in at.button if b.label == "Cancelar").click().run()

    assert not at.exception
    with db_module.get_session() as session:
        assert repository.list_entries(session) == []


def test_duplicate_date_offers_overwrite(app):
    _open_period(TODAY - timedelta(days=30))
    at = _submit(app.run(), 80.5)
    at = _submit(at, 79.0)

    assert not at.exception
    assert "Já existe um registro" in at.warning[0].value

    overwrite_button = next(b for b in at.button if b.label == "Sobrescrever registro existente")
    overwrite_button.click().run()

    assert not at.exception
    assert "79.0" in at.dataframe[0].value.to_string()


def _entry_count():
    with db_module.get_session() as session:
        return len(repository.list_entries(session))


def test_new_entry_form_starts_from_the_last_weight(app):
    _open_period(TODAY - timedelta(days=30))
    with db_module.get_session() as session:
        repository.create_entry(session, entry_date=TODAY - timedelta(days=1), weight_kg=82.3)
    at = app.run()

    assert at.number_input[0].value == pytest.approx(82.3)


def test_delete_asks_for_confirmation_first(app):
    _open_period(TODAY - timedelta(days=30))
    with db_module.get_session() as session:
        repository.create_entry(session, entry_date=TODAY - timedelta(days=1), weight_kg=82.3)
    at = app.run()

    next(b for b in at.button if b.label == "Excluir").click().run()
    assert _entry_count() == 1  # nothing deleted yet
    assert "não pode ser desfeito" in at.warning[0].value

    next(b for b in at.button if b.label == "Não").click().run()
    assert _entry_count() == 1
    assert not at.warning

    next(b for b in at.button if b.label == "Excluir").click().run()
    next(b for b in at.button if b.label == "Sim, excluir").click().run()
    assert not at.exception
    assert _entry_count() == 0
