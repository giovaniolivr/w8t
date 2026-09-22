from __future__ import annotations

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from w8t.data.models import GoalDirection, Period

_MAX_DATE = date.max


class OverlappingPeriodError(Exception):
    """Raised when a period's date range would overlap an existing one."""

    def __init__(self, other: Period):
        self.other = other
        super().__init__(f"Sobrepõe o período existente '{other.label}'.")


class PeriodNotFoundError(Exception):
    def __init__(self, period_id: int):
        self.period_id = period_id
        super().__init__(f"Período {period_id} não encontrado.")


def _overlaps(
    start_a: date, end_a: date | None, start_b: date, end_b: date | None
) -> bool:
    end_a = end_a or _MAX_DATE
    end_b = end_b or _MAX_DATE
    return start_a <= end_b and start_b <= end_a


def _find_overlap(
    session: Session, start_date: date, end_date: date | None, *, exclude_id: int | None = None
) -> Period | None:
    for other in list_periods(session):
        if exclude_id is not None and other.id == exclude_id:
            continue
        if _overlaps(start_date, end_date, other.start_date, other.end_date):
            return other
    return None


def list_periods(session: Session, *, ascending: bool = True) -> list[Period]:
    stmt = select(Period).order_by(
        Period.start_date.asc() if ascending else Period.start_date.desc()
    )
    return list(session.scalars(stmt))


def get_period(session: Session, period_id: int) -> Period:
    period = session.get(Period, period_id)
    if period is None:
        raise PeriodNotFoundError(period_id)
    return period


def get_period_for_date(session: Session, target_date: date) -> Period | None:
    for period in list_periods(session):
        if period.start_date <= target_date <= (period.end_date or _MAX_DATE):
            return period
    return None


def create_period(
    session: Session,
    *,
    label: str,
    goal_direction: GoalDirection,
    start_date: date,
    end_date: date | None = None,
    target_weight_kg: float | None = None,
) -> Period:
    overlap = _find_overlap(session, start_date, end_date)
    if overlap is not None:
        raise OverlappingPeriodError(overlap)

    period = Period(
        label=label,
        goal_direction=goal_direction,
        start_date=start_date,
        end_date=end_date,
        target_weight_kg=target_weight_kg,
    )
    session.add(period)
    session.flush()
    return period


def update_period(
    session: Session,
    period_id: int,
    *,
    label: str | None = None,
    goal_direction: GoalDirection | None = None,
    start_date: date | None = None,
    end_date: date | None = ...,  # sentinel: only overwritten when explicitly passed
    target_weight_kg: float | None = ...,  # same sentinel trick
) -> Period:
    period = get_period(session, period_id)

    new_start = start_date if start_date is not None else period.start_date
    new_end = end_date if end_date is not ... else period.end_date

    overlap = _find_overlap(session, new_start, new_end, exclude_id=period_id)
    if overlap is not None:
        raise OverlappingPeriodError(overlap)

    if label is not None:
        period.label = label
    if goal_direction is not None:
        period.goal_direction = goal_direction
    period.start_date = new_start
    period.end_date = new_end
    if target_weight_kg is not ...:
        period.target_weight_kg = target_weight_kg

    session.flush()
    return period


def delete_period(session: Session, period_id: int) -> None:
    period = get_period(session, period_id)
    session.delete(period)
    session.flush()
