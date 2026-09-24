from datetime import date, timedelta

import numpy as np
import pytest

from w8t.core import metrics
from w8t.forecasting.base import InsufficientDataError
from w8t.forecasting.kalman import KalmanSmoothTrend

D0 = date(2026, 1, 1)
Z95 = 1.959963984540054


def series_from(values, days=None):
    days = range(len(values)) if days is None else days
    return metrics.to_series(
        (D0 + timedelta(days=int(d)), v) for d, v in zip(days, values, strict=True)
    )


def simulate_smooth_trend(n, noise_sd, slope_sd, rng, level=85.0, slope=-0.05):
    """Generate data from the exact smooth-trend state-space process the model assumes."""
    y = np.empty(n)
    for t in range(n):
        y[t] = level + rng.normal(0, noise_sd)
        level, slope = level + slope, slope + rng.normal(0, slope_sd)
    return y


def with_gaps(y, rng, missing=0.15):
    keep = rng.random(len(y)) >= missing
    keep[0] = keep[-1] = True
    days = np.flatnonzero(keep)
    return series_from(y[days], days=days), days


def test_gates_on_short_span():
    with pytest.raises(InsufficientDataError):
        KalmanSmoothTrend().fit(series_from([80.0] * 14))


def test_fits_through_missing_days_without_filling_them():
    rng = np.random.default_rng(0)
    s, _ = with_gaps(85 - 0.05 * np.arange(90) + rng.normal(0, 0.3, 90), rng, missing=0.3)
    model = KalmanSmoothTrend().fit(s)
    fc = model.predict([1, 7, 30])

    assert np.isfinite(fc.mean).all()
    widths = fc.upper - fc.lower
    assert widths[0] < widths[1] < widths[2]
    # The input series still only has the real measurements.
    assert s.notna().all() and len(s) < 90


def test_recovers_measurement_noise_from_its_own_process():
    rng = np.random.default_rng(1)
    y = simulate_smooth_trend(300, noise_sd=0.35, slope_sd=0.005, rng=rng)
    model = KalmanSmoothTrend().fit(series_from(y))

    assert model.noise_sd == pytest.approx(0.35, rel=0.15)


def test_interval_is_for_a_new_measurement_not_just_the_level():
    rng = np.random.default_rng(2)
    y = simulate_smooth_trend(120, noise_sd=0.35, slope_sd=0.005, rng=rng)
    model = KalmanSmoothTrend().fit(series_from(y))
    fc = model.predict([1])

    # A 1-day-ahead measurement can't be predicted more tightly than its own noise.
    assert fc.upper[0] - fc.mean[0] >= Z95 * model.noise_sd


def test_interval_coverage_on_its_own_process_with_missing_days():
    rng = np.random.default_rng(3)
    hits = trials = 0
    for _ in range(60):
        y = simulate_smooth_trend(127, noise_sd=0.35, slope_sd=0.005, rng=rng)
        s, days = with_gaps(y[:120], rng)
        fc = KalmanSmoothTrend().fit(s).predict([7])
        truth = y[days[-1] + 7]
        hits += int(fc.lower[0] <= truth <= fc.upper[0])
        trials += 1
    assert 0.85 <= hits / trials <= 1.0


def test_boundary_optimum_is_accepted_not_discarded():
    # Perfectly steady trend + noise: the slope-variance MLE sits at ~0 (boundary), where L-BFGS
    # reports non-convergence. The Powell fallback must still produce a usable fit.
    rng = np.random.default_rng(4)
    s = series_from(85 - 0.05 * np.arange(80) + rng.normal(0, 0.3, 80))
    model = KalmanSmoothTrend().fit(s)

    assert model.slope_change_sd < 0.01
    assert np.isfinite(model.predict([7]).mean).all()
