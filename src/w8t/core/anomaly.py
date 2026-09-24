"""Baseline anomaly detection (spec item 10): robust residual vs. the recent local trend.

For each measurement, a line is fit to the *previous* measurements in a trailing calendar window
(only the past - the same rule a real-time check would have) and extrapolated to the current
date. The line is a Theil-Sen fit (median of pairwise slopes), so an outlier already inside the
window barely moves it - no need to exclude previously flagged points. (An earlier version
excluded them from an OLS fit instead; one early false positive then froze the baseline and
cascaded into a run of false flags.)

The residual is divided by the standard error of that *prediction*: noise sigma estimated
robustly from the MAD of the window's residuals (so the outliers we look for don't inflate it),
widened by the OLS prediction-interval factor sqrt(1 + 1/n + (x0 - x_mean)^2 / Sxx) because
extrapolating a slope fit on few points is itself uncertain. |z| >= ``z_threshold`` flags an
anomaly (3.5, the Iglewicz-Hoaglin convention for modified z-scores).

Measured baseline (2026-09-24, simulated trend + N(0, 0.35 kg) noise, 0.1 kg rounding, ~15%
missing days; threshold 3.5): ~1.6% of scored points falsely flagged; a +1.5 kg spike detected
~73% of the time, +2.0 kg ~100%. Raising the threshold to 4.0 trades this for ~0.9% false
positives / 60% / 87%. The dominant error source is the MAD sigma estimate from a ~15-point
window. These are the numbers a future GPR-based detector has to beat.

An anomaly means "atypical given the recent trend", not "wrong" - the measurement is never
altered or dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from w8t.core.trend import days_since

ANOMALY_WINDOW_DAYS = 21
ANOMALY_MIN_PRIOR_OBS = 8
ANOMALY_Z_THRESHOLD = 3.5
MAD_TO_SIGMA = 1.4826
# Scale floor: weights are recorded with 0.1 kg resolution, so a near-zero MAD must not turn a
# 0.2 kg difference into a huge z-score.
MIN_SCALE_KG = 0.1


def detect_anomalies(
    series: pd.Series,
    window_days: int = ANOMALY_WINDOW_DAYS,
    min_prior_obs: int = ANOMALY_MIN_PRIOR_OBS,
    z_threshold: float = ANOMALY_Z_THRESHOLD,
) -> pd.DataFrame:
    """Per-measurement ``expected_kg``, ``robust_z`` and ``is_anomaly``.

    ``expected_kg``/``robust_z`` are NaN (and ``is_anomaly`` False) where there are fewer than
    ``min_prior_obs`` usable previous measurements in the window.
    """
    expected = np.full(len(series), np.nan)
    z = np.full(len(series), np.nan)
    values = series.to_numpy()

    for i, ts in enumerate(series.index):
        in_window = (series.index < ts) & (series.index >= ts - pd.Timedelta(days=window_days))
        prior = series[in_window]
        n = len(prior)
        if n < min_prior_obs:
            continue
        days = days_since(prior.index, prior.index[0])
        if days[0] == days[-1]:
            continue
        slope, intercept, _, _ = stats.theilslopes(prior.to_numpy(), days)
        residuals = prior.to_numpy() - (intercept + slope * days)
        mad = np.median(np.abs(residuals - np.median(residuals)))
        # sqrt(n / (n - 2)): in-sample residuals of a 2-parameter fit understate the noise.
        sigma = max(MAD_TO_SIGMA * mad * (n / (n - 2)) ** 0.5, MIN_SCALE_KG)
        x0 = float((ts - prior.index[0]).days)
        sxx = float(((days - days.mean()) ** 2).sum())
        prediction_se = sigma * (1 + 1 / n + (x0 - days.mean()) ** 2 / sxx) ** 0.5

        expected[i] = intercept + slope * x0
        z[i] = (values[i] - expected[i]) / prediction_se

    flagged = np.abs(z) >= z_threshold  # NaN compares False
    return pd.DataFrame(
        {"weight_kg": values, "expected_kg": expected, "robust_z": z, "is_anomaly": flagged},
        index=series.index,
    )
