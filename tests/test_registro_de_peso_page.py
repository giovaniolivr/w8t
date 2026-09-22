from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

import w8t.data.db as db_module
from w8t.data.models import Base

PAGE_PATH = str(
    Path(__file__).resolve().parents[1] / "src" / "w8t" / "app" / "pages" / "1_Registro_de_Peso.py"
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


def test_submitting_form_creates_entry_in_history(app):
    at = app.run()

    at.number_input[0].set_value(80.5)
    at.get_by_key("FormSubmitter:new_entry_form-Salvar").click().run()

    assert not at.exception
    assert at.success[0].value.startswith("Registro de")
    assert not any("Nenhum registro" in i.value for i in at.info)


def test_duplicate_date_offers_overwrite(app):
    at = app.run()

    at.number_input[0].set_value(80.5)
    at.get_by_key("FormSubmitter:new_entry_form-Salvar").click().run()

    at.number_input[0].set_value(79.0)
    at.get_by_key("FormSubmitter:new_entry_form-Salvar").click().run()

    assert not at.exception
    assert "Já existe um registro" in at.warning[0].value

    overwrite_button = next(b for b in at.button if b.label == "Sobrescrever registro existente")
    overwrite_button.click().run()

    assert not at.exception
    assert "79.0" in at.dataframe[0].value.to_string()
