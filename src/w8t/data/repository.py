from __future__ import annotations

from datetime import date, time

from sqlalchemy import select
from sqlalchemy.orm import Session

from w8t.data.models import WeightEntry


class DuplicateEntryError(Exception):
    """Raised when trying to create a second entry for a date that already has one."""

    def __init__(self, entry_date: date):
        self.entry_date = entry_date
        super().__init__(f"Já existe um registro para {entry_date.isoformat()}.")


class EntryNotFoundError(Exception):
    """Raised when an entry id doesn't exist."""

    def __init__(self, entry_id: int):
        self.entry_id = entry_id
        super().__init__(f"Registro {entry_id} não encontrado.")


def list_entries(session: Session, *, ascending: bool = True) -> list[WeightEntry]:
    stmt = select(WeightEntry).order_by(
        WeightEntry.entry_date.asc() if ascending else WeightEntry.entry_date.desc()
    )
    return list(session.scalars(stmt))


def get_entry_by_date(session: Session, entry_date: date) -> WeightEntry | None:
    stmt = select(WeightEntry).where(WeightEntry.entry_date == entry_date)
    return session.scalars(stmt).first()


def get_entry(session: Session, entry_id: int) -> WeightEntry:
    entry = session.get(WeightEntry, entry_id)
    if entry is None:
        raise EntryNotFoundError(entry_id)
    return entry


def create_entry(
    session: Session,
    *,
    entry_date: date,
    weight_kg: float,
    entry_time: time | None = None,
) -> WeightEntry:
    if get_entry_by_date(session, entry_date) is not None:
        raise DuplicateEntryError(entry_date)

    entry = WeightEntry(entry_date=entry_date, weight_kg=weight_kg, entry_time=entry_time)
    session.add(entry)
    session.flush()
    return entry


def update_entry(
    session: Session,
    entry_id: int,
    *,
    weight_kg: float | None = None,
    entry_time: time | None = None,
) -> WeightEntry:
    entry = get_entry(session, entry_id)
    if weight_kg is not None:
        entry.weight_kg = weight_kg
    if entry_time is not None:
        entry.entry_time = entry_time
    session.flush()
    return entry


def delete_entry(session: Session, entry_id: int) -> None:
    entry = get_entry(session, entry_id)
    session.delete(entry)
    session.flush()
