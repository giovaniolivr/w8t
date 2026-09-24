from datetime import date, timedelta

import pandas as pd
import pytest

from w8t.core import metrics

D0 = date(2026, 1, 1)


def day(n: int) -> date:
    return D0 + timedelta(days=n)


def test_to_series_orders_by_entry_date_not_insertion_order():
    series = metrics.to_series([(day(2), 80.0), (day(0), 82.0), (day(1), 81.0)])

    assert list(series.index.date) == [day(0), day(1), day(2)]
    assert list(series) == [82.0, 81.0, 80.0]


def test_to_series_rejects_duplicate_dates():
    with pytest.raises(ValueError):
        metrics.to_series([(day(0), 80.0), (day(0), 81.0)])


def test_empty_series_has_no_summary():
    assert metrics.summarize(metrics.to_series([])) is None


def test_single_point_summary_reports_insufficient_derived_metrics():
    summary = metrics.summarize(metrics.to_series([(day(0), 80.0)]))

    assert summary.current_kg == summary.initial_kg == 80.0
    assert summary.change_kg == 0.0
    assert summary.pace_kg_per_week is None
    assert summary.moving_avg_7d is None
    assert summary.moving_avg_30d is None


def test_summary_basic_numbers():
    series = metrics.to_series(
        [(day(0), 90.0), (day(1), 92.0), (day(2), 88.0), (day(3), 89.0)]
    )
    summary = metrics.summarize(series)

    assert summary.n_entries == 4
    assert summary.initial_kg == 90.0
    assert summary.current_kg == 89.0
    assert (summary.min_kg, summary.min_date) == (88.0, day(2))
    assert (summary.max_kg, summary.max_date) == (92.0, day(1))
    assert summary.change_kg == pytest.approx(-1.0)
    assert summary.change_pct == pytest.approx(-1.0 / 90.0 * 100)
    assert summary.span_days == 3


def test_rolling_mean_uses_only_real_measurements_in_calendar_window():
    # Gap between day 2 and day 10: the 7-day window ending on day 10 covers days 4..10,
    # so it only sees day 10 itself -> below min_obs -> NaN (never interpolated).
    series = metrics.to_series(
        [(day(0), 80.0), (day(1), 81.0), (day(2), 82.0), (day(10), 70.0)]
    )
    ma = metrics.rolling_mean(series, 7, min_obs=3)

    assert pd.isna(ma.iloc[0])
    assert pd.isna(ma.iloc[1])
    assert ma.iloc[2] == pytest.approx(81.0)
    assert pd.isna(ma.iloc[3])


def test_rolling_mean_window_boundary_is_inclusive_of_last_7_days():
    # Window ending on day 7 is days 1..7 -> excludes day 0.
    series = metrics.to_series([(day(0), 100.0), (day(1), 80.0), (day(7), 82.0)])
    ma = metrics.rolling_mean(series, 7, min_obs=1)

    assert ma.iloc[-1] == pytest.approx(81.0)


def test_pace_exact_linear_series():
    # -0.1 kg/day -> -0.7 kg/week
    series = metrics.to_series([(day(i), 90.0 - 0.1 * i) for i in range(15)])

    assert metrics.pace_kg_per_week(series) == pytest.approx(-0.7)


def test_pace_with_irregular_spacing_uses_real_dates():
    series = metrics.to_series([(day(0), 80.0), (day(7), 81.0), (day(21), 83.0)])

    assert metrics.pace_kg_per_week(series) == pytest.approx(1.0)


def test_pace_requires_minimum_entries_and_span():
    too_few = metrics.to_series([(day(0), 80.0), (day(30), 81.0)])
    too_short = metrics.to_series([(day(i), 80.0) for i in range(5)])

    assert metrics.pace_kg_per_week(too_few) is None
    assert metrics.pace_kg_per_week(too_short) is None


def test_summary_moving_averages_at_latest_date():
    series = metrics.to_series([(day(i), 80.0 + i) for i in range(30)])
    summary = metrics.summarize(series)

    # last 7 days: 103..109 -> mean 106; last 30: 80..109 -> mean 94.5
    assert summary.moving_avg_7d == pytest.approx(106.0)
    assert summary.moving_avg_30d == pytest.approx(94.5)


def test_slice_series_is_inclusive_and_open_ended():
    series = metrics.to_series([(day(i), 80.0) for i in range(10)])

    assert len(metrics.slice_series(series, day(2), day(4))) == 3
    assert len(metrics.slice_series(series, day(5), None)) == 5
    assert len(metrics.slice_series(series, None, None)) == 10
