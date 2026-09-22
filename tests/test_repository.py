from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from w8t.data import repository
from w8t.data.models import Base


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)
    with session_local() as s:
        yield s


def test_create_and_list_entries(session):
    repository.create_entry(session, entry_date=date(2026, 1, 2), weight_kg=80.5)
    repository.create_entry(session, entry_date=date(2026, 1, 1), weight_kg=81.0)

    entries = repository.list_entries(session)

    assert [e.entry_date for e in entries] == [date(2026, 1, 1), date(2026, 1, 2)]


def test_duplicate_date_is_rejected(session):
    repository.create_entry(session, entry_date=date(2026, 1, 1), weight_kg=81.0)

    with pytest.raises(repository.DuplicateEntryError):
        repository.create_entry(session, entry_date=date(2026, 1, 1), weight_kg=82.0)


def test_update_entry_changes_weight(session):
    entry = repository.create_entry(session, entry_date=date(2026, 1, 1), weight_kg=81.0)

    updated = repository.update_entry(session, entry.id, weight_kg=79.5)

    assert updated.weight_kg == 79.5


def test_update_missing_entry_raises(session):
    with pytest.raises(repository.EntryNotFoundError):
        repository.update_entry(session, 999, weight_kg=70.0)


def test_delete_entry_removes_it(session):
    entry = repository.create_entry(session, entry_date=date(2026, 1, 1), weight_kg=81.0)

    repository.delete_entry(session, entry.id)

    assert repository.list_entries(session) == []


def test_retroactive_entry_allowed(session):
    repository.create_entry(session, entry_date=date(2020, 5, 17), weight_kg=90.0)

    entries = repository.list_entries(session)

    assert entries[0].entry_date == date(2020, 5, 17)
