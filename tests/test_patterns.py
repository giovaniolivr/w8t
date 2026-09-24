"""Trend / plateau / anomaly baselines (spec items 8-10)."""

from datetime import date, timedelta

import numpy as np
import pytest

from w8t.core import anomaly, metrics, plateau, trend
from w8t.core.trend import TrendDirection

D0 = date(2026, 1, 1)


def day(n: int) -> date:
    return D0 + timedelta(days=n)


def noisy(n_days, level, slope_per_day, sd=0.3, seed=0, missing=0.0):
    rng = np.random.default_rng(seed)
    return metrics.to_series(
        (day(d), level + slope_per_day * d + rng.normal(0, sd))
        for d in range(n_days)
        if d == n_days - 1 or rng.random() >= missing
    )


# --- linear_fit -----------------------------------------------------------------------------


def test_linear_fit_exact_line_has_zero_stderr():
    days = np.array([0.0, 1.0, 2.0, 5.0])
    fit = trend.linear_fit(days, 10.0 + 0.5 * days)

    assert fit.slope == pytest.approx(0.5)
    assert fit.intercept == pytest.approx(10.0)
    assert fit.slope_stderr == pytest.approx(0.0, abs=1e-12)
    assert fit.predict(10) == pytest.approx(15.0)


def test_linear_fit_matches_numpy_polyfit():
    rng = np.random.default_rng(1)
    days = np.sort(rng.uniform(0, 30, 20))
    weights = 80 - 0.1 * days + rng.normal(0, 0.4, 20)
    fit = trend.linear_fit(days, weights)
    (slope, intercept), cov = np.polyfit(days, weights, 1, cov="unscaled")
    resid = weights - (intercept + slope * days)

    assert fit.slope == pytest.approx(slope)
    assert fit.intercept == pytest.approx(intercept)
    assert fit.slope_stderr == pytest.approx(np.sqrt(cov[0, 0] * (resid @ resid) / 18))


def test_linear_fit_rejects_degenerate_input():
    with pytest.raises(ValueError):
        trend.linear_fit(np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        trend.linear_fit(np.array([3.0, 3.0, 3.0]), np.array([1.0, 2.0, 3.0]))


# --- trend ----------------------------------------------------------------------------------


def test_trend_insufficient_data():
    assert trend.current_trend(metrics.to_series([])) is None
    assert trend.current_trend(noisy(4, 80, 0)) is None


def test_trend_only_uses_trailing_window():
    # Steep loss long ago, flat for the last 30 days -> current trend must not see the loss.
    early = [(day(d), 90 - 0.3 * d) for d in range(30)]
    late = [(day(d), 81.0 + (0.1 if d % 2 else -0.1)) for d in range(30, 60)]
    result = trend.current_trend(metrics.to_series(early + late))

    assert result.direction is TrendDirection.STABLE
    assert result.window_start == day(59 - trend.TREND_WINDOW_DAYS + 1)


@pytest.mark.parametrize(
    ("slope_per_day", "expected"),
    [(-0.1, TrendDirection.DOWN), (0.1, TrendDirection.UP), (0.0, TrendDirection.STABLE)],
)
def test_trend_direction_on_clean_signal(slope_per_day, expected):
    result = trend.current_trend(noisy(21, 80, slope_per_day, sd=0.05))

    assert result.direction is expected
    assert result.ci_low_kg_per_week <= result.slope_kg_per_week <= result.ci_high_kg_per_week


def test_trend_undetermined_when_noise_swamps_signal():
    # 6 points, huge noise: CI is wide and straddles both zero and the stable band.
    series = metrics.to_series(
        [(day(0), 80), (day(4), 83), (day(8), 78), (day(12), 82), (day(16), 79), (day(20), 81)]
    )
    assert trend.current_trend(series).direction is TrendDirection.UNDETERMINED


# --- plateau --------------------------------------------------------------------------------


def test_no_plateau_during_steady_loss():
    assert plateau.detect_plateaus(noisy(90, 90, -0.1, seed=2)) == []


def test_plateau_detected_after_loss_phase():
    loss = [(day(d), 90 - 0.1 * d) for d in range(60)]
    rng = np.random.default_rng(3)
    flat = [(day(d), 84.0 + rng.normal(0, 0.2)) for d in range(60, 110)]
    found = plateau.detect_plateaus(metrics.to_series(loss + flat))

    assert len(found) == 1
    p = found[0]
    assert p.end == day(109)
    # Trailing windows straddling the transition blur the start - it must at least land
    # within a window of the true change point, never during the steep part.
    assert day(60) - timedelta(days=plateau.PLATEAU_WINDOW_DAYS) < p.start <= day(75)
    assert p.mean_kg == pytest.approx(84.3, abs=0.5)


def test_short_flat_stretch_is_not_a_plateau():
    series = noisy(15, 80, 0.0, sd=0.05)
    assert plateau.detect_plateaus(series) == []


# --- anomaly --------------------------------------------------------------------------------


def test_planted_anomaly_is_flagged_and_stands_out():
    series = noisy(60, 90, -0.08, sd=0.3, seed=4)
    series[series.index[40]] += 3.0
    result = anomaly.detect_anomalies(series)

    assert result["is_anomaly"].iloc[40]
    assert result["robust_z"].abs().idxmax() == series.index[40]
    # Baseline has a known ~1-2% false-positive rate (see anomaly module docstring).
    assert result["is_anomaly"].sum() <= 3


def test_anomaly_needs_prior_history():
    series = noisy(10, 80, 0.0)
    result = anomaly.detect_anomalies(series)

    assert result["robust_z"].iloc[: anomaly.ANOMALY_MIN_PRIOR_OBS].isna().all()
    assert not result["is_anomaly"].iloc[: anomaly.ANOMALY_MIN_PRIOR_OBS].any()


def test_anomaly_uses_only_past_data():
    series = noisy(40, 85, -0.05, seed=5)
    before = anomaly.detect_anomalies(series.iloc[:30])
    after = anomaly.detect_anomalies(series)

    # Appending future points must not change any past verdict.
    assert after.iloc[:30].equals(before)


def test_flagged_anomaly_does_not_contaminate_following_days():
    series = noisy(50, 85, 0.0, sd=0.2, seed=6)
    series[series.index[30]] += 5.0
    result = anomaly.detect_anomalies(series)

    assert result["is_anomaly"].iloc[30]
    assert not result["is_anomaly"].iloc[31:].any()
    assert result["expected_kg"].iloc[31] == pytest.approx(85.0, abs=0.4)


def test_anomaly_does_not_modify_the_series():
    series = noisy(40, 85, 0.0, seed=7)
    original = series.copy()
    anomaly.detect_anomalies(series)

    assert series.equals(original)
