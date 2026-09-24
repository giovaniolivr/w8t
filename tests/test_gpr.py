from datetime import date, timedelta

import numpy as np
import pytest

from w8t.core import metrics
from w8t.forecasting.base import InsufficientDataError
from w8t.forecasting.gpr import GPRLinearRBF

D0 = date(2026, 1, 1)
Z95 = 1.959963984540054


def series_from(values, days=None):
    days = range(len(values)) if days is None else days
    return metrics.to_series(
        (D0 + timedelta(days=int(d)), v) for d, v in zip(days, values, strict=True)
    )


def noisy_line(n, rng, slope=-0.05, sd=0.3, missing=0.0):
    keep = rng.random(n) >= missing
    keep[0] = keep[-1] = True
    days = np.flatnonzero(keep)
    y = 85 + slope * np.arange(n) + rng.normal(0, sd, n)
    return series_from(y[days], days=days), y


def test_gates_on_too_few_points_in_window():
    # 20 points, but spread so that fewer than 14 fall inside the trailing 60 days.
    s = series_from([80.0] * 20, days=[*range(0, 100, 10), *range(100, 200, 10)])
    with pytest.raises(InsufficientDataError):
        GPRLinearRBF(60).fit(s)


def test_uses_only_trailing_window():
    # Steep old segment, then 60 flat days -> forecast stays near the flat level.
    rng = np.random.default_rng(0)
    old = list(100 - np.arange(60.0))
    recent = list(70 + rng.normal(0, 0.2, 60))
    fc = GPRLinearRBF(60).fit(series_from(old + recent)).predict([7])

    assert fc.mean[0] == pytest.approx(70.0, abs=0.5)


def test_missing_days_are_just_absent_and_intervals_widen():
    rng = np.random.default_rng(1)
    s, _ = noisy_line(90, rng, missing=0.3)
    fc = GPRLinearRBF(60).fit(s).predict([1, 7, 30])

    assert np.isfinite(fc.mean).all()
    widths = fc.upper - fc.lower
    assert widths[0] < widths[1] < widths[2]


def test_learns_measurement_noise_and_interval_includes_it():
    rng = np.random.default_rng(2)
    s, _ = noisy_line(60, rng, sd=0.35)
    model = GPRLinearRBF(60).fit(s)
    fc = model.predict([1])

    assert model.noise_sd == pytest.approx(0.35, rel=0.3)
    assert fc.upper[0] - fc.mean[0] >= Z95 * model.noise_sd * 0.99


def test_follows_a_linear_trend_instead_of_reverting_to_the_mean():
    rng = np.random.default_rng(3)
    s, _ = noisy_line(60, rng, slope=-0.1, sd=0.2)
    fc = GPRLinearRBF(60).fit(s).predict([14])

    expected = 85 - 0.1 * (59 + 14)
    window_mean = float(s.mean())
    assert abs(fc.mean[0] - expected) < abs(fc.mean[0] - window_mean)


def test_interval_coverage_on_noisy_line_with_gaps():
    rng = np.random.default_rng(4)
    hits = trials = 0
    for _ in range(40):
        s, y = noisy_line(67, rng, sd=0.35, missing=0.15)
        s = s[s.index <= s.index[0] + np.timedelta64(59, "D")]
        fc = GPRLinearRBF(60).fit(s).predict([7])
        truth = y[(s.index[-1] - s.index[0]).days + 7]
        hits += int(fc.lower[0] <= truth <= fc.upper[0])
        trials += 1
    assert 0.85 <= hits / trials <= 1.0
