import warnings
from datetime import date, timedelta

import numpy as np
import pytest

from w8t.core import metrics
from w8t.forecasting.base import InsufficientDataError
from w8t.forecasting.holt import HoltDamped, _variance_factors, holt_filter

D0 = date(2026, 1, 1)


def series_from(values, days=None):
    days = range(len(values)) if days is None else days
    return metrics.to_series(
        (D0 + timedelta(days=int(d)), v) for d, v in zip(days, values, strict=True)
    )


def simulate_ets(n, alpha, beta, phi, sigma, rng, level=85.0, trend=-0.05):
    """Generate data from the exact ETS(A,Ad,N) process the model assumes."""
    y = np.empty(n)
    for t in range(n):
        e = rng.normal(0, sigma)
        y[t] = level + phi * trend + e
        level, trend = level + phi * trend + alpha * e, phi * trend + beta * e
    return y


def test_filter_matches_statsmodels_on_regular_series():
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    rng = np.random.default_rng(0)
    y = 80 - 0.05 * np.arange(60) + rng.normal(0, 0.3, 60)
    alpha, beta, phi, l0, b0 = 0.3, 0.05, 0.9, 80.2, -0.04

    errors, gaps, _, _ = holt_filter(y, l0, b0, alpha, beta, phi)

    # statsmodels' "known" initial states sit *before* its first observation, i.e. our day 0.
    model = ETSModel(
        y[1:], error="add", trend="add", damped_trend=True,
        initialization_method="known", initial_level=l0, initial_trend=b0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm = model.smooth([alpha, beta, phi])
    np.testing.assert_allclose(errors, y[1:] - sm.fittedvalues, atol=1e-10)
    assert (gaps == 1).all()


def test_filter_propagates_state_through_missing_days_without_imputing():
    y = np.array([80.0, np.nan, np.nan, 79.0])
    level, trend, phi = 80.0, -0.5, 0.9

    errors, gaps, _, _ = holt_filter(y, level, trend, alpha=0.5, beta=0.1, phi=phi)

    # 3 steps of pure propagation: prediction = l + (phi + phi^2 + phi^3) * b
    expected_pred = level + (phi + phi**2 + phi**3) * trend
    assert errors.tolist() == pytest.approx([79.0 - expected_pred])
    assert gaps.tolist() == [3]


def test_variance_factors():
    alpha, beta, phi = 0.3, 0.1, 0.9
    v = _variance_factors(alpha, beta, phi, 3)
    c1 = alpha + beta * phi
    c2 = alpha + beta * (phi + phi**2)

    assert v[1] == pytest.approx(1.0)
    assert v[2] == pytest.approx(1 + c1**2)
    assert v[3] == pytest.approx(1 + c1**2 + c2**2)


def test_gates_on_short_span():
    with pytest.raises(InsufficientDataError):
        HoltDamped().fit(series_from([80.0] * 14))  # 14 points but only 13 days span


def test_damped_forecast_flattens_instead_of_extrapolating():
    s = series_from(90 - 0.1 * np.arange(60))
    model = HoltDamped().fit(s)
    fc = model.predict([1, 30, 365])

    linear_365 = 90 - 0.1 * (59 + 365)
    assert fc.mean[0] < s.iloc[-1]  # still follows the recent descent
    assert fc.mean[2] > linear_365 + 10  # but a year out, nowhere near the straight line
    # the trend contribution is bounded by phi / (1 - phi) * b
    bound = model._level + model.phi / (1 - model.phi) * model._trend
    assert fc.mean[2] >= bound - 1e-9


def test_fits_with_gaps_and_intervals_widen():
    rng = np.random.default_rng(1)
    days = sorted(rng.choice(90, size=60, replace=False))
    s = series_from(85 - 0.05 * np.array(days) + rng.normal(0, 0.3, 60), days=days)
    fc = HoltDamped().fit(s).predict([1, 7, 30])

    widths = fc.upper - fc.lower
    assert np.isfinite(fc.mean).all()
    assert widths[0] < widths[1] < widths[2]


def test_recovers_parameters_roughly_from_its_own_process():
    rng = np.random.default_rng(2)
    y = simulate_ets(400, alpha=0.3, beta=0.03, phi=0.9, sigma=0.3, rng=rng)
    model = HoltDamped().fit(series_from(y))

    assert model.alpha == pytest.approx(0.3, abs=0.1)
    assert model._sigma == pytest.approx(0.3, rel=0.15)


def test_interval_coverage_on_its_own_process_with_missing_days():
    rng = np.random.default_rng(3)
    hits = trials = 0
    for _ in range(60):
        y = simulate_ets(127, alpha=0.2, beta=0.02, phi=0.95, sigma=0.35, rng=rng)
        keep = rng.random(120) >= 0.15
        keep[0] = keep[-1] = True
        days = np.flatnonzero(keep)
        fc = HoltDamped().fit(series_from(y[days], days=days)).predict([7])
        truth = y[days[-1] + 7]
        hits += int(fc.lower[0] <= truth <= fc.upper[0])
        trials += 1
    # 60 trials: binomial sd ~2.8pp around 95%; generous band, parameter-estimation noise included
    assert 0.85 <= hits / trials <= 1.0
