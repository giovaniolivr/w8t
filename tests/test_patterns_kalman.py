from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from w8t.core import metrics
from w8t.core.trend import TrendDirection
from w8t.patterns import evaluation as ev
from w8t.patterns import kalman as kal

D0 = date(2026, 1, 1)


def series_from(values, days=None):
    days = range(len(values)) if days is None else days
    return metrics.to_series(
        (D0 + timedelta(days=int(d)), float(v)) for d, v in zip(days, values, strict=True)
    )


def noisy(n, slope, seed=0, sd=0.3, missing=0.15, start=85.0):
    rng = np.random.default_rng(seed)
    keep = rng.random(n) >= missing
    keep[0] = keep[-1] = True
    days = np.flatnonzero(keep)
    return series_from(start + slope * days + rng.normal(0, sd, len(days)), days=days)


# --- insufficient data ------------------------------------------------------------------------


def test_short_series_is_reported_as_insufficient():
    s = noisy(10, -0.1)
    assert kal.smoothed_states(s) is None
    assert kal.current_trend(s) is None
    assert kal.detect_plateaus(s) == []
    flags = kal.detect_anomalies(s)
    assert not flags["is_anomaly"].any() and flags["robust_z"].isna().all()


# --- smoothed states --------------------------------------------------------------------------


def test_smoothed_states_mark_reconstructed_days():
    s = noisy(60, -0.1, missing=0.3)
    states = kal.smoothed_states(s)

    assert len(states) == 60  # daily grid
    assert states["observed"].sum() == len(s)
    assert set(states.index[states["observed"]]) == set(s.index)
    assert (states["level_lo"] <= states["level_kg"]).all()
    assert (states["slope_lo"] <= states["slope_kg_per_week"]).all()


def test_smoothed_slope_recovers_true_pace():
    states = kal.smoothed_states(noisy(120, -0.1, seed=1))
    assert states["slope_kg_per_week"].iloc[60] == pytest.approx(-0.7, abs=0.15)


# --- trend ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("slope", "expected"),
    [(-0.1, TrendDirection.DOWN), (0.1, TrendDirection.UP), (0.0, TrendDirection.STABLE)],
)
def test_trend_direction(slope, expected):
    result = kal.current_trend(noisy(90, slope, seed=2, sd=0.2))
    assert result.direction is expected
    assert result.ci_low_kg_per_week <= result.slope_kg_per_week <= result.ci_high_kg_per_week


def test_trend_uses_only_the_trailing_window():
    # Steep loss long ago, then 70 flat days -> with the 60-day window it must see "flat".
    rng = np.random.default_rng(3)
    y = np.concatenate([100 - 0.3 * np.arange(60), 82 + rng.normal(0, 0.2, 70)])
    result = kal.current_trend(series_from(y))

    assert result.direction is TrendDirection.STABLE
    assert result.window_start > D0 + timedelta(days=60)


# --- plateau ----------------------------------------------------------------------------------


def test_plateau_boundaries_track_the_true_change():
    rng = np.random.default_rng(4)
    y = np.concatenate([90 - 0.1 * np.arange(80), 82 + rng.normal(0, 0.25, 70)])
    y[:80] += rng.normal(0, 0.25, 80)
    found = kal.detect_plateaus(series_from(y))

    assert len(found) == 1
    assert abs((found[0].start - (D0 + timedelta(days=80))).days) <= 7
    assert found[0].end == D0 + timedelta(days=149)


def test_no_plateau_during_steady_gain():
    assert kal.detect_plateaus(noisy(150, 0.06, seed=5)) == []


# --- anomaly ----------------------------------------------------------------------------------


def test_planted_spike_flagged_without_follow_up_false_alarms():
    s = noisy(90, -0.08, seed=6)
    spike_day = s.index[50]
    s[spike_day] += 2.5
    flags = kal.detect_anomalies(s)

    assert flags.loc[spike_day, "is_anomaly"]
    assert flags["is_anomaly"].sum() == 1
    assert flags.loc[spike_day, "robust_z"] > kal.ANOMALY_Z_THRESHOLD


def test_anomaly_detection_does_not_modify_series():
    s = noisy(60, 0.0, seed=7)
    original = s.copy()
    kal.detect_anomalies(s)
    assert s.equals(original)


def test_same_output_columns_as_baseline():
    from w8t.core import anomaly as base

    s = noisy(60, 0.0, seed=8)
    assert list(kal.detect_anomalies(s).columns) == list(base.detect_anomalies(s).columns)


# --- labeled evaluation -----------------------------------------------------------------------


def test_labeled_series_are_deterministic_and_carry_truth():
    a = ev.labeled_series("cutting→platô", 0)
    b = ev.labeled_series("cutting→platô", 0)

    assert a.series.equals(b.series) and a.anomaly_dates == b.anomaly_dates
    assert len(a.anomaly_dates) == ev.N_ANOMALIES
    assert a.anomaly_dates <= set(a.series.index.date)
    assert len(a.true_slope_per_day) == ev.N_DAYS
    assert a.true_slope_per_day[100] == 0.0 and a.true_slope_per_day[10] == -0.08


def test_evaluate_series_counts_are_consistent():
    ls = ev.labeled_series("bulk", 1)
    rows = ev.evaluate_series(ls)
    checkpoints = len(range(ev.TREND_FIRST_CHECKPOINT, ev.N_DAYS, ev.TREND_EVERY))

    assert {r["method"] for r in rows} == {"baseline", "kalman"}
    for r in rows:
        assert r["trend_correct"] + r["trend_wrong"] + r["trend_undetermined"] == checkpoints
        assert r["anomaly_tp"] <= r["anomaly_planted"] == ev.N_ANOMALIES
    summary = ev.summarize(pd.DataFrame(rows), ["method"])
    assert summary["trend_correct"].between(0, 1).all()


# --- weekly pattern -----------------------------------------------------------------------------


def weekly_series(amp=0.5, n=120, seed=9):
    rng = np.random.default_rng(seed)
    d = np.arange(n)
    y = 80 + np.where(d % 7 < 2, amp, 0.0) + rng.normal(0, 0.3, n)
    return series_from(y)


def test_weekly_pattern_is_detected_only_when_present():
    from w8t.forecasting.kalman import fit_smooth_trend, has_weekly, weekly_effect_range

    with_pattern = fit_smooth_trend(weekly_series())
    without = fit_smooth_trend(weekly_series(amp=0.0))

    assert has_weekly(with_pattern) and not has_weekly(without)
    assert weekly_effect_range(with_pattern) == pytest.approx(0.5, abs=0.25)
    assert weekly_effect_range(without) is None


def test_weekly_pattern_is_not_tried_on_short_series():
    from w8t.forecasting.kalman import fit_smooth_trend, has_weekly

    assert not has_weekly(fit_smooth_trend(weekly_series(n=40)))


def test_detectors_handle_the_weekly_model():
    s = weekly_series()
    states = kal.smoothed_states(s)
    flags = kal.detect_anomalies(s)

    assert states["slope_kg_per_week"].abs().max() < 0.25  # flat underneath the weekly wiggle
    assert kal.current_trend(s).direction is TrendDirection.STABLE
    assert not flags["is_anomaly"].any()  # weekend bumps are expected, not anomalies
