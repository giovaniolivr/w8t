"""Baseline plateau detection (spec item 9).

A measurement date is "flat" when the OLS slope over the trailing ``window_days`` calendar days
ending on it has magnitude below ``max_abs_slope`` (kg/week), with at least ``min_obs`` points.
Consecutive flat dates are merged; a plateau spans from the first measurement inside the first
flat window to the last flat date, and is only reported if it lasts ``min_days``.

The detector is goal-agnostic: a plateau during maintenance is expected, during a cut/bulk it is
a stall - that interpretation belongs to whoever has the period context (the UI).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from w8t.core.metrics import slice_series
from w8t.core.trend import window_slope

PLATEAU_WINDOW_DAYS = 21
PLATEAU_MIN_OBS = 8
PLATEAU_MAX_ABS_SLOPE_KG_PER_WEEK = 0.25
PLATEAU_MIN_DAYS = 21


@dataclass(frozen=True)
class Plateau:
    start: date
    end: date
    n_obs: int
    mean_kg: float

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


def detect_plateaus(
    series: pd.Series,
    window_days: int = PLATEAU_WINDOW_DAYS,
    min_obs: int = PLATEAU_MIN_OBS,
    max_abs_slope: float = PLATEAU_MAX_ABS_SLOPE_KG_PER_WEEK,
    min_days: int = PLATEAU_MIN_DAYS,
) -> list[Plateau]:
    flat: list[bool] = []
    for ts in series.index:
        fit = window_slope(series, ts, window_days)
        flat.append(fit is not None and fit.n >= min_obs and abs(fit.slope * 7) < max_abs_slope)

    plateaus: list[Plateau] = []
    run_start: int | None = None
    for i, is_flat in enumerate([*flat, False]):  # sentinel closes a trailing run
        if is_flat and run_start is None:
            run_start = i
        elif not is_flat and run_start is not None:
            first_end = series.index[run_start]
            start = first_end - pd.Timedelta(days=window_days - 1)
            segment = slice_series(series, start.date(), series.index[i - 1].date())
            plateau = Plateau(
                start=segment.index[0].date(),
                end=segment.index[-1].date(),
                n_obs=len(segment),
                mean_kg=float(segment.mean()),
            )
            if plateau.days >= min_days:
                plateaus.append(plateau)
            run_start = None
    return plateaus
