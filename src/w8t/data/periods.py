from __future__ import annotations

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from w8t.data.models import GoalDirection, Period, WeightEntry

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


# --- every entry belongs to a period ---------------------------------------------------------
# Product rule (2026-09-24): a weight entry should always fall inside some period - the period's
# goal is what gives the entry analytical context (and user-defined periods are the regime
# boundaries the models respect). Association stays by date range, so "attaching" entries means
# widening a period's dates, never a foreign key.


def uncovered_entries(session: Session) -> list[WeightEntry]:
    """Entries whose date falls inside no period (e.g. logged before any period existed)."""
    all_periods = list_periods(session)
    stmt = select(WeightEntry).order_by(WeightEntry.entry_date.asc())
    return [
        e
        for e in session.scalars(stmt)
        if not any(p.start_date <= e.entry_date <= (p.end_date or _MAX_DATE) for p in all_periods)
    ]


def attachable_range(session: Session, period: Period) -> tuple[date, date | None]:
    """The (start, end) ``period`` would need to cover the uncovered entries adjacent to it -
    those between it and its neighbouring periods. Returns its current range if none."""
    others = [p for p in list_periods(session) if p.id != period.id]
    prev_end = max(
        (p.end_date for p in others if p.end_date is not None and p.end_date < period.start_date),
        default=date.min,
    )
    next_start = min(
        (p.start_date for p in others if p.start_date > (period.end_date or _MAX_DATE)),
        default=_MAX_DATE,
    )
    loose = uncovered_entries(session)
    before = [e.entry_date for e in loose if prev_end < e.entry_date < period.start_date]
    after = [
        e.entry_date
        for e in loose
        if period.end_date is not None and period.end_date < e.entry_date < next_start
    ]
    start = min(before, default=period.start_date)
    end = max(after, default=period.end_date) if period.end_date is not None else None
    return start, end


def attach_uncovered(session: Session, period_id: int) -> int:
    """Widen the period to include the uncovered entries adjacent to it. Returns how many."""
    period = get_period(session, period_id)
    before = len(uncovered_entries(session))
    start, end = attachable_range(session, period)
    if (start, end) != (period.start_date, period.end_date):
        update_period(session, period_id, start_date=start, end_date=end)
    return before - len(uncovered_entries(session))


def start_including_uncovered(session: Session, entry_date: date) -> date:
    """Start date for a *new* period created for ``entry_date``: reaches back to the earliest
    uncovered entry since the last period before it, so earlier loose entries get included."""
    prev_end = max(
        (p.end_date for p in list_periods(session)
         if p.end_date is not None and p.end_date < entry_date),
        default=date.min,
    )
    earlier = [e.entry_date for e in uncovered_entries(session)
               if prev_end < e.entry_date <= entry_date]
    return min([*earlier, entry_date])


def period_containing(all_periods: list[Period], target: date) -> Period | None:
    """Pure lookup (no session): the period whose range contains ``target``."""
    return next(
        (p for p in all_periods if p.start_date <= target <= (p.end_date or _MAX_DATE)), None
    )
