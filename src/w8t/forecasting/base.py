"""Pluggable forecasting interface.

Every model follows the same contract so the backtesting layer can treat them uniformly:

- ``fit(series)`` - reestimate parameters from a weight series indexed by ``entry_date``
  (irregular spacing allowed, gaps never filled). "Training" here means refitting a light model
  on every call, not training a network.
- ``predict(horizons, level)`` - point forecast **and** prediction interval for each horizon,
  measured in calendar days after the last observation. There is no point-only method: a
  forecast without uncertainty is never produced.
- Each model self-gates: ``fit`` raises :class:`InsufficientDataError` (with a human-readable
  reason) when the series is too short for that model, instead of returning a fragile result.

Models only ever see the series they are given - callers (backtesting) are responsible for
passing only data available at the forecast origin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


class InsufficientDataError(Exception):
    """The series doesn't have enough data for this model."""


class NotFittedError(Exception):
    pass


@dataclass(frozen=True)
class Forecast:
    """Future values - never to be stored alongside or confused with real measurements."""

    model: str
    origin: pd.Timestamp  # last observation date used to fit
    horizons: np.ndarray  # calendar days after origin
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    level: float

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.origin + pd.to_timedelta(self.horizons, unit="D")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"horizon_days": self.horizons, "mean": self.mean, "lower": self.lower,
             "upper": self.upper},
            index=pd.DatetimeIndex(self.dates, name="date"),
        )


class ForecastModel(ABC):
    name: str
    min_obs: int

    def __init__(self) -> None:
        self._origin: pd.Timestamp | None = None

    def fit(self, series: pd.Series) -> ForecastModel:
        if len(series) < self.min_obs:
            raise InsufficientDataError(
                f"{self.name}: precisa de ≥ {self.min_obs} medições, há {len(series)}."
            )
        if not series.index.is_monotonic_increasing:
            raise ValueError("Série precisa estar ordenada por entry_date.")
        self._fit(series)
        self._origin = series.index[-1]
        return self

    def predict(self, horizons: Sequence[int], level: float = 0.95) -> Forecast:
        if self._origin is None:
            raise NotFittedError(f"{self.name}: chame fit() antes de predict().")
        h = np.asarray(horizons, dtype=int)
        if (h < 1).any():
            raise ValueError("Horizontes devem ser ≥ 1 dia.")
        if not 0 < level < 1:
            raise ValueError("level deve estar entre 0 e 1.")
        mean, lower, upper = self._predict(h, level)
        return Forecast(self.name, self._origin, h, mean, lower, upper, level)

    @abstractmethod
    def _fit(self, series: pd.Series) -> None: ...

    @abstractmethod
    def _predict(
        self, horizons: np.ndarray, level: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
