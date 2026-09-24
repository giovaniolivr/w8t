"""Trend / plateau / anomaly detection from the Kalman smooth-trend model.

Model-based counterparts of the deterministic baselines in ``w8t.core`` (``trend``, ``plateau``,
``anomaly``), which remain the reference: these only earn a place if the labeled evaluation in
``w8t.patterns.evaluation`` shows them better. Same output types as the baselines, so the UI can
swap them.

Which pass of the Kalman model each one uses, and why:
- Current trend -> *filtered* slope at the last day (equals the smoothed one there): only past,
  fitted on the trailing ``TREND_WINDOW_DAYS`` only. Fitted on the whole history, the slope
  variance is estimated tiny (most of the history has a steady pace) and the filter reacts slowly
  after a regime change; a 60-day window keeps it responsive. Labeled evaluation, 48 series
  (docs/patterns_benchmark.md): correct 74% / wrong 3.7% with 60 days vs. 73% / 4.5% with all
  history and 59% / 3.8% for the 21-day OLS baseline; chosen among {all, 90, 60} days on that
  same evaluation.
- Plateaus -> *smoothed* slope (RTS, uses the whole series). Plateau detection is descriptive -
  "where was the weight flat?" - so using later data to locate a past plateau is legitimate and
  is exactly what removes the trailing-window lag of the baseline. Never used for evaluation of
  forecasts.
- Anomalies -> *filter* innovations (one-step-ahead prediction errors, past states only),
  standardized by their own variance. A flagged point is masked (treated as missing) and the
  model refit, so it can't drag the state and trigger follow-up false alarms; largest |z| first.
  The noise/slope variances are estimated on the whole series (global hyperparameters).

The model's state on a *missing* day is a reconstruction, not a measurement - nothing here
writes or returns it as one.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from w8t.core.plateau import PLATEAU_MIN_DAYS, Plateau
from w8t.core.trend import STABLE_BAND_KG_PER_WEEK, Trend, TrendDirection
from w8t.forecasting.kalman import fit_smooth_trend, weekly_effect_range

TREND_WINDOW_DAYS = 60
MIN_OBS = 14
MIN_SPAN_DAYS = 21
ANOMALY_Z_THRESHOLD = 3.5
ANOMALY_MAX_ROUNDS = 20
Z95 = 1.959963984540054


def _sd(variance):
    # With sigma2.trend estimated at 0 the slope variance collapses and rounding can leave it at
    # about -1e-18; clip instead of letting sqrt produce NaN.
    return np.sqrt(np.clip(variance, 0.0, None))


def _enough(series: pd.Series) -> bool:
    return len(series) >= MIN_OBS and (series.index[-1] - series.index[0]).days >= MIN_SPAN_DAYS


def smoothed_states(series: pd.Series) -> pd.DataFrame | None:
    """Daily smoothed level and slope (kg/week) with 95% bands; ``observed`` marks real days.

    ``None`` if the series is too short. Rows where ``observed`` is False are model
    reconstructions and must be presented as such.
    """
    if not _enough(series):
        return None
    result = fit_smooth_trend(series)
    # States: level, slope, then (if the weekly pattern was detected) the seasonal states.
    level, slope = result.smoothed_state[0], result.smoothed_state[1]
    level_sd = _sd(result.smoothed_state_cov[0, 0])
    slope_sd = _sd(result.smoothed_state_cov[1, 1])
    index = pd.date_range(series.index[0], series.index[-1], freq="D", name="date")
    states = pd.DataFrame(
        {
            "observed": index.isin(series.index),
            "level_kg": level,
            "level_lo": level - Z95 * level_sd,
            "level_hi": level + Z95 * level_sd,
            "slope_kg_per_week": slope * 7,
            "slope_lo": (slope - Z95 * slope_sd) * 7,
            "slope_hi": (slope + Z95 * slope_sd) * 7,
        },
        index=index,
    )
    # Level is the deseasonalized weight; when a weekly pattern was detected, record its size so
    # the UI can say the trend line excludes it.
    states.attrs["weekly_range_kg"] = weekly_effect_range(result)
    return states


def _classify(slope: float, low: float, high: float, band: float) -> TrendDirection:
    if high < 0 and slope <= -band:
        return TrendDirection.DOWN
    if low > 0 and slope >= band:
        return TrendDirection.UP
    if -band <= low and high <= band:
        return TrendDirection.STABLE
    return TrendDirection.UNDETERMINED


def current_trend(
    series: pd.Series,
    band: float = STABLE_BAND_KG_PER_WEEK,
    window_days: int | None = TREND_WINDOW_DAYS,
) -> Trend | None:
    """Direction from the filtered slope at the latest day, same decision rule as the baseline
    (direction only if the 95% CI excludes zero and |slope| >= band; stable if the CI sits
    inside the band; otherwise undetermined)."""
    if window_days is not None and not series.empty:
        series = series[series.index > series.index[-1] - pd.Timedelta(days=window_days)]
    if series.empty or not _enough(series):
        return None
    result = fit_smooth_trend(series)
    slope = float(result.filtered_state[1, -1]) * 7
    half = Z95 * float(_sd(result.filtered_state_cov[1, 1, -1])) * 7
    low, high = slope - half, slope + half
    return Trend(
        direction=_classify(slope, low, high, band),
        slope_kg_per_week=slope,
        ci_low_kg_per_week=low,
        ci_high_kg_per_week=high,
        window_start=series.index[0].date(),
        window_end=series.index[-1].date(),
        n_obs=len(series),
    )


def detect_plateaus(
    series: pd.Series,
    max_abs_slope: float = STABLE_BAND_KG_PER_WEEK,
    min_days: int = PLATEAU_MIN_DAYS,
) -> list[Plateau]:
    """Stretches of >= ``min_days`` where the smoothed slope stays within ±``max_abs_slope``."""
    states = smoothed_states(series)
    if states is None:
        return []
    flat = (states["slope_kg_per_week"].abs() < max_abs_slope).to_numpy()
    plateaus = []
    start = None
    for i, is_flat in enumerate([*flat, False]):
        if is_flat and start is None:
            start = i
        elif not is_flat and start is not None:
            first, last = states.index[start].date(), states.index[i - 1].date()
            segment = series[(series.index.date >= first) & (series.index.date <= last)]
            if (last - first).days + 1 >= min_days and len(segment):
                plateaus.append(
                    Plateau(
                        start=segment.index[0].date(),
                        end=segment.index[-1].date(),
                        n_obs=len(segment),
                        mean_kg=float(segment.mean()),
                    )
                )
            start = None
    return plateaus


def detect_anomalies(
    series: pd.Series,
    z_threshold: float = ANOMALY_Z_THRESHOLD,
    max_rounds: int = ANOMALY_MAX_ROUNDS,
) -> pd.DataFrame:
    """Same columns as ``w8t.core.anomaly.detect_anomalies``: weight_kg, expected_kg (one-step
    filter prediction), robust_z (standardized innovation), is_anomaly."""
    empty = pd.DataFrame(
        {"weight_kg": series.to_numpy(), "expected_kg": np.nan, "robust_z": np.nan,
         "is_anomaly": False},
        index=series.index,
    )
    if not _enough(series):
        return empty

    masked: set[date] = set()
    result = None
    for _ in range(max_rounds + 1):
        working = series[~np.isin(series.index.date, list(masked))]
        result = fit_smooth_trend(working.reindex(series.index))
        predicted, z = _one_step(result, series)
        candidates = np.abs(z) >= z_threshold
        candidates &= ~np.isin(series.index.date, list(masked))
        if not candidates.any() or len(masked) >= max_rounds:
            break
        worst = int(np.nanargmax(np.where(candidates, np.abs(z), -np.inf)))
        masked.add(series.index[worst].date())

    predicted, z = _one_step(result, series)
    out = empty.copy()
    out["expected_kg"] = predicted
    out["robust_z"] = z
    out["is_anomaly"] = np.isin(series.index.date, list(masked))
    return out


def _one_step(result, series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """One-step-ahead prediction and standardized error at each *measured* date. Masked dates
    are NaN in the fitted data, but their prediction still exists, so their z is computed
    against the real value. The diffuse-initialization steps are left out (z = NaN)."""
    fr = result.filter_results
    grid = pd.date_range(series.index[0], series.index[-1], freq="D")
    pos = grid.get_indexer(series.index)
    predicted = fr.forecasts[0, pos]
    sd = np.sqrt(fr.forecasts_error_cov[0, 0, pos])
    z = (series.to_numpy() - predicted) / sd
    z[pos < result.loglikelihood_burn] = np.nan
    return predicted, z
