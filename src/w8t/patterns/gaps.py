"""Gap reconstruction: estimate what the scale would likely have read on days with no entry.

Opt-in only (the user picks a gap). The result is category 3 of the value taxonomy -
*reconstructed/interpolated* - so it is returned as a separate frame, never written to
``weight_entries`` and never mixed with measurements.

Each method returns, per missing day, a point estimate and a 95% interval **for a measurement**
(underlying level uncertainty + scale noise), so it can be checked against held-out real
measurements:

- ``linear``: straight line between the two measurements flanking the gap. Interval from the
  scale noise estimated on consecutive-day differences, with the variance of interpolating two
  noisy endpoints. Ignores any curvature - the reference to beat.
- ``kalman``: RTS-smoothed level of the smooth-trend model fitted on the whole series (uses data
  on both sides of the gap, which is the point here), variance = smoothed level variance +
  measurement noise.
- ``gpr``: GPR (linear + RBF + white noise, the forecasting kernel) fitted on measurements within
  ``GPR_CONTEXT_DAYS`` of the gap on both sides; predictive std includes the learned noise.

Only interior gaps (measurements on both sides) are reconstructed - after the last entry it would
be a forecast, which is the forecasting layer's job.

Before any method runs, measurements flagged by the Kalman anomaly detector are set aside (they
are not deleted - only not used as evidence here). Atypical spikes inflate the estimated scale
noise: on the labeled evaluation, Kalman intervals covered 98% of held-out measurements (too
wide) with them and 94.8% (target 95%) without, 20% narrower, same error.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor

from w8t.forecasting.gpr import _kernel as gpr_kernel
from w8t.forecasting.kalman import fit_smooth_trend
from w8t.patterns.kalman import _enough, detect_anomalies

METHODS = ("linear", "kalman", "gpr")
GPR_CONTEXT_DAYS = 45
MIN_CONTEXT_OBS = 8
Z95 = float(stats.norm.ppf(0.975))


class CannotReconstructError(Exception):
    """Not enough data around/overall to reconstruct this gap with the chosen method."""


@dataclass(frozen=True)
class Gap:
    start: date  # first missing day
    end: date  # last missing day

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def dates(self) -> list[date]:
        return [self.start + timedelta(days=i) for i in range(self.days)]


def find_gaps(series: pd.Series, min_days: int = 1) -> list[Gap]:
    """Interior runs of consecutive days without a measurement, at least ``min_days`` long."""
    gaps = []
    for before, after in zip(series.index[:-1], series.index[1:], strict=True):
        missing = (after - before).days - 1
        if missing >= min_days:
            gaps.append(
                Gap((before + pd.Timedelta(days=1)).date(), (after - pd.Timedelta(days=1)).date())
            )
    return gaps


def reconstruct(
    series: pd.Series, gap: Gap, method: str = "kalman", *, mask_anomalies: bool = True
) -> pd.DataFrame:
    """Per missing day: ``estimate_kg``, ``lower``, ``upper`` (95% for a measurement)."""
    if method not in METHODS:
        raise ValueError(f"Método desconhecido: {method}")
    if not (series.index[0].date() < gap.start and gap.end < series.index[-1].date()):
        raise CannotReconstructError("Só lacunas internas (com registros antes e depois).")
    if mask_anomalies:
        series = without_anomalies(series)
    days = pd.DatetimeIndex([pd.Timestamp(d) for d in gap.dates()], name="date")
    mean, sd = {"linear": _linear, "kalman": _kalman, "gpr": _gpr}[method](series, days)
    return pd.DataFrame(
        {"estimate_kg": mean, "lower": mean - Z95 * sd, "upper": mean + Z95 * sd}, index=days
    )


def without_anomalies(series: pd.Series) -> pd.Series:
    """The series minus points the Kalman detector flags (unchanged if too short to run it)."""
    flags = detect_anomalies(series)
    return series[~flags["is_anomaly"].to_numpy()]


def _noise_sd(series: pd.Series) -> float:
    consecutive = series.diff()[series.index.to_series().diff().dt.days == 1].dropna()
    if len(consecutive) < 5:
        raise CannotReconstructError("Poucos pares de dias consecutivos para estimar o ruído.")
    return float(consecutive.std(ddof=1) / np.sqrt(2))


def _linear(series: pd.Series, days: pd.DatetimeIndex):
    before = series[series.index < days[0]]
    after = series[series.index > days[-1]]
    t0, y0 = before.index[-1], before.iloc[-1]
    t1, y1 = after.index[0], after.iloc[0]
    w = ((days - t0).days / (t1 - t0).days).to_numpy(dtype=float)
    mean = y0 + w * (y1 - y0)
    s = _noise_sd(series)
    return mean, s * np.sqrt(1 + w**2 + (1 - w) ** 2)


def _kalman(series: pd.Series, days: pd.DatetimeIndex):
    if not _enough(series):
        raise CannotReconstructError("Histórico curto demais para o modelo de Kalman.")
    result = fit_smooth_trend(series)
    grid = pd.date_range(series.index[0], series.index[-1], freq="D")
    pos = grid.get_indexer(days)
    level = result.smoothed_state[0, pos]
    level_var = np.clip(result.smoothed_state_cov[0, 0, pos], 0.0, None)
    params = dict(zip(result.model.param_names, np.asarray(result.params), strict=True))
    noise_var = float(params["sigma2.irregular"])
    return level, np.sqrt(level_var + noise_var)


def _gpr(series: pd.Series, days: pd.DatetimeIndex):
    lo = days[0] - pd.Timedelta(days=GPR_CONTEXT_DAYS)
    hi = days[-1] + pd.Timedelta(days=GPR_CONTEXT_DAYS)
    context = series[(series.index >= lo) & (series.index <= hi)]
    if len(context) < MIN_CONTEXT_OBS:
        raise CannotReconstructError("Poucas medições em volta da lacuna para o GPR.")
    x = (context.index - days[0]).days.to_numpy(dtype=float).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp = GaussianProcessRegressor(
            gpr_kernel(), normalize_y=True, n_restarts_optimizer=1, random_state=0
        ).fit(x, context.to_numpy())
    x_new = (days - days[0]).days.to_numpy(dtype=float).reshape(-1, 1)
    mean, sd = gp.predict(x_new, return_std=True)
    return mean, sd
