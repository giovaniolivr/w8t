from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from w8t.config import settings
from w8t.core import metrics
from w8t.data import demo, periods, repository
from w8t.data.models import Base

TODAY = date(2026, 9, 24)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_generate_series_is_deterministic_and_ends_today():
    a = demo.generate_series(TODAY)

    assert a == demo.generate_series(TODAY)
    assert a[-1][0] == TODAY
    assert len({d for d, _ in a}) == len(a)


def test_generate_series_has_gaps_and_the_planted_anomaly():
    points = demo.generate_series(TODAY)
    series = metrics.to_series(points)

    assert len(points) < demo.N_DAYS  # some days are missing
    residual = series - metrics.rolling_mean(series, 7, min_obs=1)
    anomaly_date = TODAY - timedelta(days=demo.N_DAYS - 1 - demo.ANOMALY_DAY)
    assert residual.idxmax().date() == anomaly_date


def test_reset_refuses_outside_demo_mode(session, monkeypatch):
    monkeypatch.setattr(settings, "app_env", "local")
    repository.create_entry(session, entry_date=TODAY, weight_kg=80.0)

    with pytest.raises(demo.NotDemoModeError):
        demo.reset_demo_data(session, today=TODAY)
    assert len(repository.list_entries(session)) == 1


def test_reset_replaces_existing_data(session, monkeypatch):
    monkeypatch.setattr(settings, "app_env", "demo")
    repository.create_entry(session, entry_date=date(2020, 1, 1), weight_kg=50.0)

    n = demo.reset_demo_data(session, today=TODAY)
    entries = repository.list_entries(session)

    assert n == len(entries) == len(demo.generate_series(TODAY))
    assert entries[0].entry_date != date(2020, 1, 1)
    assert [p.label for p in periods.list_periods(session)] == ["Cutting", "Manutenção"]
