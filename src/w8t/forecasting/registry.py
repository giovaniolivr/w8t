"""All forecasting models available to backtesting and the UI, simplest first."""

from __future__ import annotations

from w8t.forecasting.base import ForecastModel
from w8t.forecasting.baselines import baseline_models
from w8t.forecasting.holt import HoltDamped


def all_models() -> list[ForecastModel]:
    return [*baseline_models(), HoltDamped()]
