"""All forecasting models available to backtesting and the UI, simplest first."""

from __future__ import annotations

from w8t.forecasting.base import ForecastModel
from w8t.forecasting.baselines import baseline_models
from w8t.forecasting.ensemble import EqualWeightEnsemble
from w8t.forecasting.gpr import GPR
from w8t.forecasting.holt import HoltDamped
from w8t.forecasting.kalman import KalmanSmoothTrend


def kalman_holt_ensemble() -> EqualWeightEnsemble:
    return EqualWeightEnsemble(
        [KalmanSmoothTrend(), HoltDamped()], interval="mixture", name="Combinação Kalman + Holt"
    )


def all_models() -> list[ForecastModel]:
    return [
        *baseline_models(),
        HoltDamped(),
        KalmanSmoothTrend(),
        # Chosen on held-out seeds, confirmed (not significantly) on the benchmark: see gpr.py.
        GPR("linear + Matérn 3/2", 90),
        kalman_holt_ensemble(),
    ]
