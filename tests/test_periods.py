from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from w8t.data import periods
from w8t.data.models import Base, GoalDirection


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)
    with session_local() as s:
        yield s


def test_create_and_list_periods(session):
    periods.create_period(
        session,
        label="Cutting verão",
        goal_direction=GoalDirection.LOSS,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
    )
    periods.create_period(
        session,
        label="Bulk pós-verão",
        goal_direction=GoalDirection.GAIN,
        start_date=date(2026, 4, 1),
    )

    result = periods.list_periods(session)

    assert [p.label for p in result] == ["Cutting verão", "Bulk pós-verão"]
    assert result[1].end_date is None  # período em andamento


def test_overlapping_period_is_rejected(session):
    periods.create_period(
        session,
        label="Cutting verão",
        goal_direction=GoalDirection.LOSS,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
    )

    with pytest.raises(periods.OverlappingPeriodError):
        periods.create_period(
            session,
            label="Manutenção",
            goal_direction=GoalDirection.MAINTENANCE,
            start_date=date(2026, 3, 1),
            end_date=date(2026, 6, 1),
        )


def test_open_ended_period_blocks_any_later_start(session):
    periods.create_period(
        session,
        label="Bulk",
        goal_direction=GoalDirection.GAIN,
        start_date=date(2026, 1, 1),
    )

    with pytest.raises(periods.OverlappingPeriodError):
        periods.create_period(
            session,
            label="Cutting",
            goal_direction=GoalDirection.LOSS,
            start_date=date(2027, 1, 1),
        )


def test_adjacent_non_overlapping_periods_allowed(session):
    periods.create_period(
        session,
        label="Cutting",
        goal_direction=GoalDirection.LOSS,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
    )

    period = periods.create_period(
        session,
        label="Manutenção",
        goal_direction=GoalDirection.MAINTENANCE,
        start_date=date(2026, 4, 1),
        end_date=date(2026, 6, 30),
    )

    assert period.label == "Manutenção"


def test_update_period_excludes_itself_from_overlap_check(session):
    period = periods.create_period(
        session,
        label="Cutting",
        goal_direction=GoalDirection.LOSS,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
    )

    updated = periods.update_period(session, period.id, label="Cutting (renomeado)")

    assert updated.label == "Cutting (renomeado)"
    assert updated.start_date == date(2026, 1, 1)


def test_update_period_end_date_to_none_reopens_it(session):
    period = periods.create_period(
        session,
        label="Cutting",
        goal_direction=GoalDirection.LOSS,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
    )

    updated = periods.update_period(session, period.id, end_date=None)

    assert updated.end_date is None


def test_update_period_still_rejects_new_overlap(session):
    periods.create_period(
        session,
        label="Cutting",
        goal_direction=GoalDirection.LOSS,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
    )
    manutencao = periods.create_period(
        session,
        label="Manutenção",
        goal_direction=GoalDirection.MAINTENANCE,
        start_date=date(2026, 4, 1),
        end_date=date(2026, 6, 30),
    )

    with pytest.raises(periods.OverlappingPeriodError):
        periods.update_period(session, manutencao.id, start_date=date(2026, 3, 15))


def test_get_period_for_date(session):
    periods.create_period(
        session,
        label="Cutting",
        goal_direction=GoalDirection.LOSS,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
    )

    assert periods.get_period_for_date(session, date(2026, 2, 15)).label == "Cutting"
    assert periods.get_period_for_date(session, date(2025, 12, 31)) is None
    assert periods.get_period_for_date(session, date(2026, 4, 1)) is None


def test_delete_period(session):
    period = periods.create_period(
        session,
        label="Cutting",
        goal_direction=GoalDirection.LOSS,
        start_date=date(2026, 1, 1),
    )

    periods.delete_period(session, period.id)

    assert periods.list_periods(session) == []


def test_update_missing_period_raises(session):
    with pytest.raises(periods.PeriodNotFoundError):
        periods.update_period(session, 999, label="x")
