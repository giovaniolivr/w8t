from __future__ import annotations

from datetime import UTC, date, datetime, time

from sqlalchemy import Date, DateTime, Float, Time, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


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
