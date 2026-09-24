from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from w8t.core import metrics
from w8t.patterns import evaluation as ev
from w8t.patterns import gaps
from w8t.patterns import gaps_evaluation as gev

D0 = date(2026, 1, 1)


def series_from(values, days):
    return metrics.to_series(
        (D0 + timedelta(days=int(d)), float(v)) for d, v in zip(days, values, strict=True)
    )


def noisy_line_with_hole(hole=range(40, 50), n=90, slope=-0.1, sd=0.3, seed=0):
    rng = np.random.default_rng(seed)
    days = [d for d in range(n) if d not in hole]
    y = 85 + slope * np.array(days) + rng.normal(0, sd, len(days))
    return series_from(y, days)


def test_find_gaps_interior_only_with_min_length():
    s = series_from([80.0] * 6, days=[0, 1, 3, 4, 9, 10])
    found = gaps.find_gaps(s)

    assert [(g.start, g.end, g.days) for g in found] == [
        (D0 + timedelta(days=2), D0 + timedelta(days=2), 1),
        (D0 + timedelta(days=5), D0 + timedelta(days=8), 4),
    ]
    assert [g.days for g in gaps.find_gaps(s, min_days=2)] == [4]


def test_rejects_edge_gap_and_unknown_method():
    s = noisy_line_with_hole()
    outside = gaps.Gap(D0 - timedelta(days=3), D0 - timedelta(days=1))
    with pytest.raises(gaps.CannotReconstructError):
        gaps.reconstruct(s, outside)
    with pytest.raises(ValueError):
        gaps.reconstruct(s, gaps.find_gaps(s)[0], method="magic")


def test_linear_interpolates_between_flanking_measurements():
    days = list(range(10)) + list(range(14, 30))
    y = [80.0 + (0.05 if d % 2 else -0.05) for d in days]
    y[9], y[10] = 80.0, 82.0  # flanks of the 4-day hole (days 10..13)
    s = series_from(y, days)
    rec = gaps.reconstruct(s, gaps.find_gaps(s)[0], method="linear", mask_anomalies=False)

    np.testing.assert_allclose(rec["estimate_kg"], [80.4, 80.8, 81.2, 81.6])
    assert (rec["lower"] < rec["estimate_kg"]).all() and (rec["estimate_kg"] < rec["upper"]).all()


@pytest.mark.parametrize("method", ["kalman", "gpr"])
def test_model_methods_recover_underlying_trend(method):
    s = noisy_line_with_hole()
    rec = gaps.reconstruct(s, gaps.find_gaps(s)[0], method=method)
    truth = 85 - 0.1 * np.arange(40, 50)

    assert np.abs(rec["estimate_kg"].to_numpy() - truth).mean() < 0.3
    assert ((rec["lower"] <= truth) & (truth <= rec["upper"])).all()


def test_output_covers_exactly_the_missing_days_and_input_is_untouched():
    s = noisy_line_with_hole()
    original = s.copy()
    gap = gaps.find_gaps(s)[0]
    rec = gaps.reconstruct(s, gap)

    assert list(rec.index.date) == gap.dates()
    assert not set(rec.index) & set(s.index)  # never a measured day
    assert s.equals(original)


def test_anomalous_flank_is_not_used_as_evidence():
    s = noisy_line_with_hole(seed=1)
    flank = s.index[s.index < pd.Timestamp(D0 + timedelta(days=40))][-1]
    s[flank] += 4.0  # a wild reading right before the gap
    gap = gaps.find_gaps(s)[0]

    naive = gaps.reconstruct(s, gap, method="linear", mask_anomalies=False)
    masked = gaps.reconstruct(s, gap, method="linear")
    truth_start = 85 - 0.1 * 40
    assert abs(masked["estimate_kg"].iloc[0] - truth_start) < 1.0
    assert naive["estimate_kg"].iloc[0] - truth_start > 2.0


def test_gap_evaluation_rows():
    rows = gev.evaluate_series(ev.labeled_series("bulk", 0))
    df = pd.DataFrame(rows)

    assert set(df["method"]) == set(gaps.METHODS)
    assert not df["failed"].any()
    assert set(df["hole_days"]) == set(gev.HOLE_LENGTHS)
    summary = gev.summarize(df, ["method"])
    assert summary["coverage"].between(0, 1).all()
