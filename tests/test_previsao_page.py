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
from w8t.forecasting.registry import all_models, kalman_holt_ensemble

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
    # Too short for the recommended ensemble (Kalman/Holt need 14 points) -> falls back to the
    # first model that can fit, and still forecasts with an interval.
    assert at.selectbox[0].value != kalman_holt_ensemble().name
    assert "Intervalo 95% (kg)" in at.dataframe[-1].value.columns


def test_backtest_table_and_best_model_preselected(app):
    _seed(90)
    at = app.run()

    assert not at.exception
    evaluation = at.dataframe[0].value
    assert set(evaluation["Modelo"]) == {m.name for m in all_models()}
    # The benchmark-recommended ensemble is the default; the lowest-MAE model is labelled.
    assert at.selectbox[0].value == kalman_holt_ensemble().name
    best = evaluation.groupby("Modelo")["MAE (kg)"].mean().idxmin()
    if best != kalman_holt_ensemble().name:
        assert f"{best} (menor erro neste histórico)" in at.selectbox[0].options
    assert any("cobertura real foi" in c.value for c in at.caption)


def test_model_that_cannot_fit_says_so(app):
    _seed(4)  # below every model's minimum (naive needs 5)
    at = app.run()

    assert not at.exception
    assert any("não pode prever" in i.value for i in at.info)


def test_significance_table_is_shown(app):
    _seed(90)
    at = app.run()

    assert not at.exception
    table = at.dataframe[1].value
    assert list(table.columns) == [
        "Horizonte (dias)", "Menor MAE", "Comparado com", "Casos independentes", "p (Holm)",
        "Conclusão",
    ]
    # ~90 days of history: 30-day comparisons can't have enough independent cases.
    h30 = table[table["Horizonte (dias)"] == 30]
    assert (h30["Conclusão"] == "não testável: poucos casos independentes").all()
