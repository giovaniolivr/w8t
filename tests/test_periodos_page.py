from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

import w8t.data.db as db_module
from w8t.data.models import Base

PAGE_PATH = str(
    Path(__file__).resolve().parents[1] / "src" / "w8t" / "app" / "pages" / "2_Periodos.py"
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
