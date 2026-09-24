"""Baseline trend detection (spec item 8): OLS slope over a trailing calendar window.

The direction is only called when the data supports it: the slope's 95% confidence interval must
exclude zero *and* the slope must exceed a practical-relevance band. When the interval sits
entirely inside the band the series is confidently stable; anything else is "indefinido" (too
noisy / too few points to tell) - never forced into a direction.

This is the deliberately simple baseline the future GPR-based version will be compared against.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from w8t.core.metrics import slice_series

TREND_WINDOW_DAYS = 21
TREND_MIN_OBS = 6
# |slope| below this (kg/week) is treated as practically flat.
STABLE_BAND_KG_PER_WEEK = 0.25
CONFIDENCE = 0.95


class TrendDirection(str, enum.Enum):
    DOWN = "descendo"
    UP = "subindo"
    STABLE = "estável"
    UNDETERMINED = "indefinido"


@dataclass(frozen=True)
class LinearFit:
    """Least-squares line over (days since first point, weight). Slopes in kg/day."""

    slope: float
    intercept: float
    slope_stderr: float
    n: int

    def predict(self, day: float) -> float:
        return self.intercept + self.slope * day


def linear_fit(days: np.ndarray, weights: np.ndarray) -> LinearFit:
    """OLS fit with the slope's standard error. Requires >= 3 points and >= 2 distinct days."""
    n = len(days)
    if n < 3:
        raise ValueError("linear_fit precisa de pelo menos 3 pontos.")
    x_mean = days.mean()
    sxx = float(((days - x_mean) ** 2).sum())
    if sxx == 0:
        raise ValueError("linear_fit precisa de pelo menos 2 datas distintas.")
    slope = float(((days - x_mean) * (weights - weights.mean())).sum() / sxx)
    intercept = float(weights.mean() - slope * x_mean)
    residuals = weights - (intercept + slope * days)
    sigma2 = float((residuals**2).sum() / (n - 2))
    return LinearFit(slope, intercept, (sigma2 / sxx) ** 0.5, n)


def days_since(index: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    return (index - origin).days.to_numpy(dtype=float)


@dataclass(frozen=True)
class Trend:
    direction: TrendDirection
    slope_kg_per_week: float
    ci_low_kg_per_week: float
    ci_high_kg_per_week: float
    window_start: date
    window_end: date
    n_obs: int


def window_slope(series: pd.Series, end: pd.Timestamp, window_days: int) -> LinearFit | None:
    """OLS over the measurements in the ``window_days`` calendar days ending on ``end``."""
    window = slice_series(series, (end - pd.Timedelta(days=window_days - 1)).date(), end.date())
    if len(window) < 3 or window.index[0] == window.index[-1]:
        return None
    return linear_fit(days_since(window.index, window.index[0]), window.to_numpy())


def classify(
    fit: LinearFit, band: float = STABLE_BAND_KG_PER_WEEK, confidence: float = CONFIDENCE
) -> tuple[TrendDirection, float, float]:
    """Direction plus the slope's confidence interval, both in kg/week."""
    t_crit = stats.t.ppf(0.5 + confidence / 2, df=fit.n - 2)
    slope, half = fit.slope * 7, t_crit * fit.slope_stderr * 7
    low, high = slope - half, slope + half

    if high < 0 and slope <= -band:
        direction = TrendDirection.DOWN
    elif low > 0 and slope >= band:
        direction = TrendDirection.UP
    elif -band <= low and high <= band:
        direction = TrendDirection.STABLE
    else:
        direction = TrendDirection.UNDETERMINED
    return direction, low, high


def current_trend(
    series: pd.Series,
    window_days: int = TREND_WINDOW_DAYS,
    min_obs: int = TREND_MIN_OBS,
) -> Trend | None:
    """Trend over the window ending at the latest measurement. ``None`` if data is insufficient."""
    if series.empty:
        return None
    end = series.index[-1]
    fit = window_slope(series, end, window_days)
    if fit is None or fit.n < min_obs:
        return None

    direction, low, high = classify(fit)
    return Trend(
        direction=direction,
        slope_kg_per_week=fit.slope * 7,
        ci_low_kg_per_week=low,
        ci_high_kg_per_week=high,
        window_start=(end - pd.Timedelta(days=window_days - 1)).date(),
        window_end=end.date(),
        n_obs=fit.n,
    )
