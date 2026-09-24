"""Synthetic dataset for the public demo deployment (APP_ENV=demo).

Plausible ~6-month trajectory with the features the analytics layers must deal with: a
downward trend, day-to-day noise, a plateau, one deliberate anomaly and missing days. Seeded, so
every reset produces the same shape; dates are anchored to ``today`` so the demo always looks
current.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
from sqlalchemy import delete
from sqlalchemy.orm import Session

from w8t.config import settings
from w8t.data.models import GoalDirection, Period, WeightEntry

SEED = 42
N_DAYS = 180
START_KG = 88.0
CUT_DAYS = 90  # linear loss phase
PLATEAU_DAYS = 30  # still "cutting", but the weight stalls
CUT_RATE_KG_PER_DAY = -0.08
NOISE_SD_KG = 0.35
MISSING_RATE = 0.15
ANOMALY_DAY = 60
ANOMALY_KG = 2.5


class NotDemoModeError(RuntimeError):
    """Resetting wipes every entry - only ever allowed against the demo database."""


def generate_series(today: date) -> list[tuple[date, float]]:
    rng = np.random.default_rng(SEED)
    start = today - timedelta(days=N_DAYS - 1)

    days = np.arange(N_DAYS)
    trend = START_KG + CUT_RATE_KG_PER_DAY * np.minimum(days, CUT_DAYS)
    after_plateau = np.clip(days - (CUT_DAYS + PLATEAU_DAYS), 0, None)
    trend = trend + 0.01 * after_plateau  # maintenance: slight regain
    weights = trend + rng.normal(0, NOISE_SD_KG, N_DAYS)
    weights[ANOMALY_DAY] += ANOMALY_KG

    observed = rng.random(N_DAYS) >= MISSING_RATE
    observed[[0, ANOMALY_DAY, N_DAYS - 1]] = True

    return [
        (start + timedelta(days=int(d)), round(float(w), 1))
        for d, w, keep in zip(days, weights, observed, strict=True)
        if keep
    ]


def reset_demo_data(session: Session, *, today: date) -> int:
    """Replace all entries and periods with the synthetic dataset. Returns the entry count."""
    if not settings.is_demo:
        raise NotDemoModeError("Reset só é permitido com APP_ENV=demo.")

    session.execute(delete(WeightEntry))
    session.execute(delete(Period))

    points = generate_series(today)
    session.add_all(WeightEntry(entry_date=d, weight_kg=w) for d, w in points)

    start = today - timedelta(days=N_DAYS - 1)
    maintenance_start = start + timedelta(days=CUT_DAYS + PLATEAU_DAYS)
    session.add_all(
        [
            Period(
                label="Cutting",
                goal_direction=GoalDirection.LOSS,
                start_date=start,
                end_date=maintenance_start - timedelta(days=1),
                target_weight_kg=80.0,
            ),
            Period(
                label="Manutenção",
                goal_direction=GoalDirection.MAINTENANCE,
                start_date=maintenance_start,
            ),
        ]
    )
    session.flush()
    return len(points)
