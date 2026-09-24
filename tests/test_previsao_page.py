from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

import w8t.data.db as db_module
from w8t.data import repository
from w8t.data.models import Base

TODAY = date.today()  # noqa: DTZ011 - entries are relative to "today" like real usage
PAGE_PATH = str(
    Path(__file__).resolve().parents[1] / "src" / "w8t" / "app" / "pages" / "3_Previsao.py"
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
    return AppTest.from_file(PAGE_PATH, default_timeout=30)


def _seed(n_days, slope=-0.1, sd=0.2, seed=0):
    rng = np.random.default_rng(seed)
    start = TODAY - timedelta(days=n_days - 1)
    with db_module.get_session() as session:
        for i in range(n_days):
            repository.create_entry(
                session,
                entry_date=start + timedelta(days=i),
                weight_kg=round(90 + slope * i + rng.normal(0, sd), 1),
            )


def test_empty_state(app):
    at = app.run()
    assert not at.exception
    assert "Nenhum registro" in at.info[0].value


def test_short_history_explains_why_no_evaluation(app):
    _seed(10)
    at = app.run()

    assert not at.exception
    assert "histórico suficiente para avaliar" in at.info[0].value
    # Linear model needs 7 points in 28 days -> 10 points is enough to forecast, with interval.
    assert "Intervalo 95% (kg)" in at.dataframe[-1].value.columns


def test_backtest_table_and_best_model_preselected(app):
    _seed(90)
    at = app.run()

    assert not at.exception
    evaluation = at.dataframe[0].value
    assert set(evaluation["Modelo"]) == {"Último valor", "Média móvel 7d", "Regressão linear 28d"}
    # Clean linear trend -> regression has the lowest backtest error and is suggested first.
    assert at.selectbox[0].value == "Regressão linear 28d"
    assert any("cobertura real foi" in c.value for c in at.caption)


def test_model_that_cannot_fit_says_so(app):
    _seed(4)  # below every model's minimum (naive needs 5)
    at = app.run()

    assert not at.exception
    assert any("não pode prever" in i.value for i in at.info)
