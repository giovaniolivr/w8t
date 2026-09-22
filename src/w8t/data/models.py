from __future__ import annotations

import enum
from datetime import UTC, date, datetime, time

from sqlalchemy import Date, DateTime, Enum, Float, String, Time, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class GoalDirection(str, enum.Enum):
    """Controlled vocabulary driving analysis; the period's `label` carries the user's own
    wording (e.g. "cutting de verão")."""

    LOSS = "loss"
    GAIN = "gain"
    MAINTENANCE = "maintenance"


class WeightEntry(Base):
    """One weight measurement per calendar day (spec item 1)."""

    __tablename__ = "weight_entries"
    __table_args__ = (UniqueConstraint("entry_date", name="uq_weight_entries_entry_date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    entry_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    weight_kg: Mapped[float] = mapped_column(Float, nullable=False)
    entry_time: Mapped[time | None] = mapped_column(Time, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )


class Period(Base):
    """A bulking/cutting/maintenance cycle (or any user-defined stretch of time).

    Optional by design - entries are associated to a period by date-range containment at
    query time, not by a foreign key, so period boundaries can be edited freely without
    touching WeightEntry rows, and entries outside any period simply have none.
    """

    __tablename__ = "periods"

    id: Mapped[int] = mapped_column(primary_key=True)
    label: Mapped[str] = mapped_column(String, nullable=False)
    goal_direction: Mapped[GoalDirection] = mapped_column(Enum(GoalDirection), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    target_weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )
