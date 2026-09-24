from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from w8t.core import metrics
from w8t.forecasting.base import InsufficientDataError, NotFittedError
from w8t.forecasting.baselines import (
    LinearTrend,
    MovingAverage,
    NaiveLastValue,
    baseline_models,
)

D0 = date(2026, 1, 1)
Z95 = 1.959963984540054


def series_from(values, days=None):
    days = range(len(values)) if days is None else days
    return metrics.to_series((D0 + timedelta(days=d), v) for d, v in zip(days, values, strict=True))


# --- interface ------------------------------------------------------------------------------


@pytest.mark.parametrize("model", baseline_models(), ids=lambda m: m.name)
def test_predict_before_fit_raises(model):
    with pytest.raises(NotFittedError):
        model.predict([1])


@pytest.mark.parametrize("model", baseline_models(), ids=lambda m: m.name)
def test_models_self_gate_on_short_series(model):
    with pytest.raises(InsufficientDataError):
        model.fit(series_from([80.0, 80.1, 80.2]))


@pytest.mark.parametrize("model", baseline_models(), ids=lambda m: m.name)
def test_forecast_shape_dates_and_interval_order(model):
    rng = np.random.default_rng(0)
    s = series_from(80 - 0.05 * np.arange(40) + rng.normal(0, 0.3, 40))
    fc = model.fit(s).predict([1, 7, 30])

    assert list(fc.dates) == [s.index[-1] + pd.Timedelta(days=h) for h in (1, 7, 30)]
    assert (fc.lower < fc.mean).all() and (fc.mean < fc.upper).all()
    assert fc.level == 0.95
    assert list(fc.to_frame().columns) == ["horizon_days", "mean", "lower", "upper"]


def test_invalid_horizon_and_level():
    model = NaiveLastValue().fit(series_from([80.0, 81, 80, 81, 80]))
    with pytest.raises(ValueError):
        model.predict([0])
    with pytest.raises(ValueError):
        model.predict([1], level=1.0)


def test_unsorted_series_rejected():
    s = series_from([80.0, 81, 80, 81, 80]).iloc[::-1]
    with pytest.raises(ValueError):
        NaiveLastValue().fit(s)


# --- naive ----------------------------------------------------------------------------------


def test_naive_mean_and_random_walk_interval():
    fc = NaiveLastValue().fit(series_from([80.0, 81, 80, 81, 80])).predict([1, 4])

    np.testing.assert_allclose(fc.mean, [80.0, 80.0])
    # sigma per day = 1 -> half-widths z*1*sqrt(1), z*1*sqrt(4)
    np.testing.assert_allclose(fc.upper - fc.mean, [Z95, 2 * Z95])


def test_naive_scales_differences_by_day_gap():
    # Each step is 2 kg over 4 days -> (dy^2 / dt) = 1 -> sigma per day = 1.
    s = series_from([80.0, 82, 80, 82, 80], days=[0, 4, 8, 12, 16])
    fc = NaiveLastValue().fit(s).predict([1])

    assert fc.upper[0] - fc.mean[0] == pytest.approx(Z95)


# --- moving average -------------------------------------------------------------------------


def test_moving_average_level_is_trailing_window_mean():
    values = [90.0] * 20 + [80.0, 81.0, 82.0]
    fc = MovingAverage(3).fit(series_from(values)).predict([1, 10])

    np.testing.assert_allclose(fc.mean, [81.0, 81.0])
    # constant width across horizons (the model's stated naive assumption)
    assert fc.upper[0] - fc.lower[0] == pytest.approx(fc.upper[1] - fc.lower[1])


def test_moving_average_gates_on_empty_recent_window():
    # Plenty of history, but nothing in the last 7 days before the origin except the origin.
    s = series_from([80.0] * 15 + [80.0], days=[*range(15), 40])
    with pytest.raises(InsufficientDataError):
        MovingAverage(7).fit(s)


# --- linear trend ---------------------------------------------------------------------------


def test_linear_trend_exact_line():
    s = series_from(90 - 0.1 * np.arange(30))
    fc = LinearTrend(28).fit(s).predict([1, 10])

    np.testing.assert_allclose(fc.mean, [90 - 0.1 * 30, 90 - 0.1 * 39])
    np.testing.assert_allclose(fc.upper - fc.lower, [0, 0], atol=1e-9)


def test_linear_trend_uses_only_trailing_window():
    # Steep old segment, flat recent 28 days -> forecast must be flat.
    s = series_from(list(100 - np.arange(30.0)) + [70.0] * 28)
    fc = LinearTrend(28).fit(s).predict([7])

    assert fc.mean[0] == pytest.approx(70.0)


def test_linear_trend_interval_widens_with_horizon():
    rng = np.random.default_rng(1)
    s = series_from(80 - 0.05 * np.arange(28) + rng.normal(0, 0.3, 28))
    fc = LinearTrend(28).fit(s).predict([1, 7, 30])

    widths = fc.upper - fc.lower
    assert widths[0] < widths[1] < widths[2]


def test_linear_trend_interval_coverage_when_assumptions_hold():
    # Truly linear + Gaussian noise: the 95% interval must cover ~95% of new observations.
    rng = np.random.default_rng(2)
    hits = trials = 0
    for _ in range(400):
        y = 85 - 0.07 * np.arange(35) + rng.normal(0, 0.35, 35)
        fc = LinearTrend(28).fit(series_from(y[:28])).predict([1, 7])
        truth = y[[28, 34]]
        hits += int(((fc.lower <= truth) & (truth <= fc.upper)).sum())
        trials += 2
    assert 0.93 <= hits / trials <= 0.97
