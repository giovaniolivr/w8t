"""Descriptive dashboard metrics (spec item 2) - deterministic, no ML.

Everything here works on *real measurements only*: a gap in the series stays a gap. Rolling
means are time-based windows ("7D"/"30D") over whatever measurements fall inside them - no
interpolation, no resampling to daily frequency. Each metric that needs a minimum amount of data
returns ``None`` when there isn't enough, instead of a number computed from too few points.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

# Minimum number of real measurements inside the window for a rolling mean to be reported.
ROLLING_MIN_OBS = {7: 3, 30: 8}

# Pace (kg/week) is a least-squares slope; with fewer points or a very short span it is noise.
PACE_MIN_ENTRIES = 3
PACE_MIN_SPAN_DAYS = 7


def to_series(points: Iterable[tuple[date, float]]) -> pd.Series:
    """Build a weight series indexed by ``entry_date``, sorted chronologically.

    Insertion order is irrelevant - the series is always ordered by the date the weight refers
    to. Duplicate dates are rejected (the data layer already guarantees one entry per day).
    """
    points = list(points)
    if not points:
        return pd.Series([], index=pd.DatetimeIndex([], name="entry_date"), dtype=float,
                         name="weight_kg")

    dates, weights = zip(*points, strict=True)
    index = pd.DatetimeIndex(pd.to_datetime(list(dates)), name="entry_date")
    series = pd.Series(list(weights), index=index, dtype=float, name="weight_kg").sort_index()
    if series.index.has_duplicates:
        raise ValueError("Série contém mais de uma medição para a mesma data.")
    return series


def slice_series(series: pd.Series, start: date | None, end: date | None) -> pd.Series:
    """Restrict the series to ``[start, end]`` (inclusive). ``None`` means unbounded."""
    mask = np.ones(len(series), dtype=bool)
    if start is not None:
        mask &= series.index >= pd.Timestamp(start)
    if end is not None:
        mask &= series.index <= pd.Timestamp(end)
    return series[mask]


def rolling_mean(series: pd.Series, window_days: int, min_obs: int | None = None) -> pd.Series:
    """Trailing time-based rolling mean evaluated at each real measurement date.

    The window covers the ``window_days`` calendar days ending on (and including) each date.
    Points where the window holds fewer than ``min_obs`` measurements are NaN.
    """
    if min_obs is None:
        min_obs = ROLLING_MIN_OBS.get(window_days, max(1, window_days // 2))
    if series.empty:
        return series.copy()
    return series.rolling(f"{window_days}D", min_periods=min_obs).mean()


def pace_kg_per_week(series: pd.Series) -> float | None:
    """Least-squares slope of weight vs. time, in kg/week. ``None`` if data is insufficient."""
    if len(series) < PACE_MIN_ENTRIES:
        return None
    days = (series.index - series.index[0]).days.to_numpy(dtype=float)
    if days[-1] < PACE_MIN_SPAN_DAYS:
        return None
    slope_per_day, _ = np.polyfit(days, series.to_numpy(), 1)
    return float(slope_per_day * 7)


@dataclass(frozen=True)
class Summary:
    n_entries: int
    first_date: date
    last_date: date
    initial_kg: float
    current_kg: float
    min_kg: float
    min_date: date
    max_kg: float
    max_date: date
    change_kg: float
    change_pct: float
    pace_kg_per_week: float | None
    moving_avg_7d: float | None
    moving_avg_30d: float | None

    @property
    def span_days(self) -> int:
        return (self.last_date - self.first_date).days


def summarize(series: pd.Series) -> Summary | None:
    """Headline numbers for a (possibly scoped) series. ``None`` if the series is empty.

    "Current" is the latest measurement by ``entry_date``; the moving averages reported are the
    values at that latest date (``None`` when the trailing window lacks enough measurements).
    """
    if series.empty:
        return None

    initial = float(series.iloc[0])
    current = float(series.iloc[-1])
    ma7 = rolling_mean(series, 7).iloc[-1]
    ma30 = rolling_mean(series, 30).iloc[-1]

    return Summary(
        n_entries=len(series),
        first_date=series.index[0].date(),
        last_date=series.index[-1].date(),
        initial_kg=initial,
        current_kg=current,
        min_kg=float(series.min()),
        min_date=series.idxmin().date(),
        max_kg=float(series.max()),
        max_date=series.idxmax().date(),
        change_kg=current - initial,
        change_pct=(current - initial) / initial * 100,
        pace_kg_per_week=pace_kg_per_week(series),
        moving_avg_7d=None if pd.isna(ma7) else float(ma7),
        moving_avg_30d=None if pd.isna(ma30) else float(ma30),
    )
