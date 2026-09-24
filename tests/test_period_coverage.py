from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from w8t.core.trend import Trend, TrendDirection
from w8t.data import periods, repository
from w8t.data.models import Base, GoalDirection, Period
from w8t.patterns.regime import suggested_goal

D0 = date(2026, 3, 1)


def d(n: int) -> date:
    return D0 + timedelta(days=n)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def add_entries(session, days):
    for n in days:
        repository.create_entry(session, entry_date=d(n), weight_kg=80.0)


def new_period(session, start, end=None, goal=GoalDirection.LOSS, label="P"):
    return periods.create_period(
        session, label=label, goal_direction=goal, start_date=d(start),
        end_date=None if end is None else d(end),
    )


def test_uncovered_entries(session):
    add_entries(session, [0, 1, 2, 10, 11])
    new_period(session, 10)

    assert [e.entry_date for e in periods.uncovered_entries(session)] == [d(0), d(1), d(2)]


def test_attach_extends_open_period_back_over_loose_entries(session):
    # The user's case: entries logged first, period created later starting after them.
    add_entries(session, [0, 1, 2, 3, 4, 5, 6])
    p = new_period(session, 7)

    assert periods.attach_uncovered(session, p.id) == 7
    assert p.start_date == d(0)
    assert periods.uncovered_entries(session) == []


def test_attach_never_crosses_another_period(session):
    add_entries(session, [0, 5, 6, 20, 25])
    new_period(session, 0, 4, label="antigo")
    p = new_period(session, 10, 15, label="atual")

    assert periods.attachable_range(session, p) == (d(5), d(25))
    assert periods.attach_uncovered(session, p.id) == 4  # days 5, 6, 20, 25
    assert periods.uncovered_entries(session) == []


def test_attach_is_a_noop_without_adjacent_loose_entries(session):
    add_entries(session, [10, 11])
    p = new_period(session, 10)

    assert periods.attach_uncovered(session, p.id) == 0
    assert p.start_date == d(10)


def test_new_period_start_reaches_back_to_loose_entries_since_last_period(session):
    add_entries(session, [0, 1, 8, 9])
    new_period(session, 0, 2)

    assert periods.start_including_uncovered(session, d(12)) == d(8)
    assert periods.start_including_uncovered(session, d(1)) == d(1)  # covered already


def _trend(direction):
    return Trend(direction, 0.5, 0.2, 0.8, d(0), d(30), 20)


@pytest.mark.parametrize(
    ("goal", "direction", "expected"),
    [
        (GoalDirection.LOSS, TrendDirection.UP, GoalDirection.GAIN),
        (GoalDirection.GAIN, TrendDirection.DOWN, GoalDirection.LOSS),
        (GoalDirection.MAINTENANCE, TrendDirection.UP, GoalDirection.GAIN),
        (GoalDirection.MAINTENANCE, TrendDirection.DOWN, GoalDirection.LOSS),
        (GoalDirection.LOSS, TrendDirection.DOWN, None),  # on track
        (GoalDirection.LOSS, TrendDirection.STABLE, None),  # a stall is not a new phase
        (GoalDirection.GAIN, TrendDirection.UNDETERMINED, None),  # not enough evidence
    ],
)
def test_suggested_goal_for_indefinite_periods(goal, direction, expected):
    period = Period(label="p", goal_direction=goal, start_date=d(0), end_date=None)
    assert suggested_goal(period, _trend(direction)) == expected


def test_no_suggestion_for_planned_periods_or_missing_data():
    planned = Period(label="p", goal_direction=GoalDirection.LOSS, start_date=d(0),
                     end_date=d(60))
    assert suggested_goal(planned, _trend(TrendDirection.UP)) is None
    assert suggested_goal(None, _trend(TrendDirection.UP)) is None
    open_ = Period(label="p", goal_direction=GoalDirection.LOSS, start_date=d(0), end_date=None)
    assert suggested_goal(open_, None) is None
