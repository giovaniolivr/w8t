from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

import w8t.data.db as db_module
from w8t.data import repository
from w8t.data.models import Base, WeightEntry

TODAY = date.today()  # noqa: DTZ011 - entries are relative to "today" like real usage
PAGE_PATH = str(
    Path(__file__).resolve().parents[1] / "src" / "w8t" / "app" / "views" / "4_Lacunas.py"
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


def _seed(days, slope=-0.1, seed=0):
    rng = np.random.default_rng(seed)
    start = TODAY - timedelta(days=max(days))
    with db_module.get_session() as session:
        for d in days:
            repository.create_entry(
                session, entry_date=start + timedelta(days=d),
                weight_kg=round(85 + slope * d + rng.normal(0, 0.3), 1),
            )


def _count_entries():
    with db_module.get_session() as session:
        return session.scalar(select(func.count()).select_from(WeightEntry))


def test_no_gaps(app):
    _seed(range(30))
    at = app.run()

    assert not at.exception
    assert "Nenhuma lacuna" in at.info[0].value


def test_reconstruction_is_opt_in_shown_as_estimate_and_never_saved(app):
    days = [d for d in range(80) if not 40 <= d < 46]
    _seed(days)
    before = _count_entries()

    at = app.run()
    assert not at.exception
    assert len(at.dataframe) == 0  # nothing estimated until the user asks

    at.button[0].click().run()
    assert not at.exception
    table = at.dataframe[0].value
    assert len(table) == 6
    assert (table["Tipo"] == "estimativa — não é medição").all()
    assert _count_entries() == before  # estimates never touch weight_entries


def test_method_that_cannot_run_explains_why(app):
    _seed([0, 1, 2, 5, 6, 7])  # far too short for the Kalman model
    at = app.run()
    at.button[0].click().run()

    assert not at.exception
    assert "Não foi possível estimar" in at.info[0].value
