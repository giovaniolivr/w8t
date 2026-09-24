"""Which detectors run: Kalman-based ones (better on the labeled evaluation, see
docs/patterns_benchmark.md), with the deterministic baselines from ``w8t.core`` as fallback when
the history is too short for the model. Shared by the dashboard and the insights summary so both
always describe the same results."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from w8t.core import anomaly, plateau, trend
from w8t.core.plateau import Plateau
from w8t.core.trend import Trend
from w8t.patterns import kalman as kal


@dataclass(frozen=True)
class Patterns:
    method: str  # "kalman" | "baseline" (plateaus and anomalies)
    trend: Trend | None
    trend_method: str  # "kalman" | "baseline"
    plateaus: list[Plateau]
    anomalies: pd.DataFrame  # weight_kg, expected_kg, robust_z, is_anomaly
    states: pd.DataFrame | None  # Kalman smoothed states, None on the baseline path


def detect(series: pd.Series) -> Patterns:
    states = kal.smoothed_states(series)
    if states is None:
        return Patterns(
            "baseline", trend.current_trend(series), "baseline",
            plateau.detect_plateaus(series), anomaly.detect_anomalies(series), None,
        )
    k_trend = kal.current_trend(series)
    return Patterns(
        "kalman",
        k_trend if k_trend is not None else trend.current_trend(series),
        "kalman" if k_trend is not None else "baseline",
        kal.detect_plateaus(series),
        kal.detect_anomalies(series),
        states,
    )
