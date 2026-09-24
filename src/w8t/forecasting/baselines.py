"""Baseline forecasters (roadmap steps 1-2): last value, moving average, windowed linear trend.

They exist to be beaten. Each one's prediction interval is exactly what its own (simple)
assumptions imply - including where those assumptions are knowingly naive - so the backtesting
layer can measure interval coverage and show *where* each baseline is over/under-confident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from w8t.core.metrics import slice_series
from w8t.core.trend import LinearFit, days_since, linear_fit
from w8t.forecasting.base import ForecastModel, InsufficientDataError


def _z(level: float) -> float:
    return float(stats.norm.ppf(0.5 + level / 2))


class NaiveLastValue(ForecastModel):
    """Tomorrow looks like today: random walk.

    Point forecast = last measurement. Variance grows linearly with the horizon,
    ``sigma^2 * h``, with ``sigma^2`` (per day) estimated from consecutive differences scaled by
    their day gap, ``(dy)^2 / dt`` - so irregular spacing is accounted for, not interpolated.
    """

    name = "Último valor"
    min_obs = 5

    def _fit(self, series: pd.Series) -> None:
        dy = np.diff(series.to_numpy())
        dt = np.diff(days_since(series.index, series.index[0]))
        self._last = float(series.iloc[-1])
        self._sigma_per_day = float(np.sqrt(np.mean(dy**2 / dt)))

    def _predict(self, horizons, level):
        mean = np.full(len(horizons), self._last)
        half = _z(level) * self._sigma_per_day * np.sqrt(horizons)
        return mean, mean - half, mean + half


class MovingAverage(ForecastModel):
    """Flat forecast at the mean of the last ``window_days`` calendar days.

    Interval: the standard deviation of this model's own *past one-step-ahead errors* (each
    trailing-window mean vs. the next measurement, past-only), constant across horizons. That
    constancy is the model's naive assumption (a level that doesn't drift) - backtesting will
    show it under-covering at long horizons whenever there is a trend.
    """

    min_obs = 10
    min_window_obs = 3
    error_history_days = 90

    def __init__(self, window_days: int = 7) -> None:
        super().__init__()
        self.window_days = window_days
        self.name = f"Média móvel {window_days}d"

    def _window_mean(self, series: pd.Series, end: pd.Timestamp) -> float | None:
        start = end - pd.Timedelta(days=self.window_days - 1)
        window = slice_series(series, start.date(), end.date())
        return float(window.mean()) if len(window) >= self.min_window_obs else None

    def _fit(self, series: pd.Series) -> None:
        level = self._window_mean(series, series.index[-1])
        if level is None:
            raise InsufficientDataError(
                f"{self.name}: precisa de ≥ {self.min_window_obs} medições nos últimos "
                f"{self.window_days} dias."
            )
        cutoff = series.index[-1] - pd.Timedelta(days=self.error_history_days)
        history = series[series.index > cutoff]
        errors = []
        for prev_ts, y in zip(history.index[:-1], history.iloc[1:], strict=True):
            m = self._window_mean(series, prev_ts)
            if m is not None:
                errors.append(y - m)
        if len(errors) < 5:
            raise InsufficientDataError(
                f"{self.name}: histórico insuficiente para estimar a incerteza."
            )
        self._level = level
        self._sd = float(np.std(errors, ddof=1))

    def _predict(self, horizons, level):
        mean = np.full(len(horizons), self._level)
        half = np.full(len(horizons), _z(level) * self._sd)
        return mean, mean - half, mean + half


class LinearTrend(ForecastModel):
    """OLS line over the last ``window_days`` calendar days, extrapolated.

    Interval: classical OLS prediction interval (Student t), which already widens with distance
    from the window's center. It assumes the trend continues unchanged - the assumption that
    breaks at plateaus and phase changes.
    """

    min_obs = 7

    def __init__(self, window_days: int = 28) -> None:
        super().__init__()
        self.window_days = window_days
        self.name = f"Regressão linear {window_days}d"

    def _fit(self, series: pd.Series) -> None:
        end = series.index[-1]
        window = slice_series(
            series, (end - pd.Timedelta(days=self.window_days - 1)).date(), end.date()
        )
        if len(window) < self.min_obs:
            raise InsufficientDataError(
                f"{self.name}: precisa de ≥ {self.min_obs} medições nos últimos "
                f"{self.window_days} dias, há {len(window)}."
            )
        self._window_start = window.index[0]
        self._fit_result: LinearFit = linear_fit(
            days_since(window.index, self._window_start), window.to_numpy()
        )
        self._origin_day = float((end - self._window_start).days)

    def _predict(self, horizons, level):
        fit = self._fit_result
        x = self._origin_day + horizons
        mean = fit.predict(x)
        se = np.array([fit.prediction_se(d) for d in x])
        half = float(stats.t.ppf(0.5 + level / 2, df=fit.n - 2)) * se
        return mean, mean - half, mean + half


def baseline_models() -> list[ForecastModel]:
    return [NaiveLastValue(), MovingAverage(7), LinearTrend(28)]
