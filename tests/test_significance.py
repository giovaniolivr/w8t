import numpy as np
import pandas as pd
import pytest
from scipy import stats

from w8t.forecasting.significance import _holm, compare, dm_test, versus_best


def forecasts_frame(errors_by_model: dict[str, np.ndarray], horizon: int = 7) -> pd.DataFrame:
    n = len(next(iter(errors_by_model.values())))
    origins = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.concat(
        pd.DataFrame({"model": name, "origin": origins, "horizon": horizon, "error": e})
        for name, e in errors_by_model.items()
    )


def ma_noise(n, lags, rng):
    """MA(lags) noise - the dependence structure overlapping h-step forecasts produce."""
    eps = rng.normal(0, 1, n + lags)
    return np.convolve(eps, np.ones(lags + 1), mode="valid")[:n]


def test_clear_winner_is_significant():
    rng = np.random.default_rng(0)
    fc = forecasts_frame({"good": rng.normal(0, 0.3, 80), "bad": rng.normal(0, 1.0, 80)})
    result = compare(fc, "good", "bad", 7, step_days=1)

    assert result.mean_loss_diff < 0
    assert result.p_value < 0.001
    assert result.lags == 6
    assert result.effective_cases == pytest.approx(80 / 7)


def test_too_few_effectively_independent_cases_is_not_tested():
    # 80 daily origins at h=30 overlap so much they're worth < 3 independent cases.
    rng = np.random.default_rng(0)
    fc = forecasts_frame(
        {"good": rng.normal(0, 0.3, 80), "bad": rng.normal(0, 1.0, 80)}, horizon=30
    )
    assert compare(fc, "good", "bad", 30, step_days=1) is None
    table = versus_best(fc, step_days=1)
    assert not table["testable"].any()
    assert table["p_value"].isna().all()


def test_identical_losses_give_nan_not_a_fake_result():
    e = np.random.default_rng(1).normal(0, 0.5, 30)
    statistic, p_value = dm_test(np.abs(e) - np.abs(e), lags=0)

    assert np.isnan(statistic) and np.isnan(p_value)


def test_too_few_common_cases_returns_none():
    fc = forecasts_frame({"a": np.ones(5), "b": np.ones(5)})
    assert compare(fc, "a", "b", 7, step_days=1) is None


def test_false_positive_rate_under_overlap_is_roughly_controlled():
    # Equal-accuracy models with MA(6)-dependent loss differentials (h=7, step=1).
    # DM is known to be somewhat liberal here (~7-8% measured); a naive paired t-test is far
    # worse - this is why the correction matters.
    rng = np.random.default_rng(2)
    rejections = naive_rejections = 0
    trials = 600
    for _ in range(trials):
        d = ma_noise(90, 6, rng)
        _, p = dm_test(d, lags=6)
        rejections += p < 0.05
        naive_rejections += stats.ttest_1samp(d, 0).pvalue < 0.05
    assert rejections / trials < 0.11
    assert naive_rejections / trials > 0.25


def test_versus_best_picks_lowest_mae_and_adjusts_p_values():
    rng = np.random.default_rng(3)
    fc = forecasts_frame(
        {
            "best": rng.normal(0, 0.3, 60),
            "close": rng.normal(0, 0.45, 60),
            "far": rng.normal(0, 1.5, 60),
        },
        horizon=1,  # no overlap: 60 effective cases (at h=7 they'd be only 60/7 < 10)
    )
    table = versus_best(fc, step_days=1).set_index("other")

    assert set(table.index) == {"close", "far"}
    assert table["testable"].all()
    assert (table["best"] == "best").all()
    assert table.loc["far", "p_holm"] < 0.001
    assert (table["p_holm"] >= table["p_value"]).all()


def test_holm_adjustment():
    adjusted = _holm([0.01, 0.04, 0.03, float("nan")])

    assert adjusted[0] == pytest.approx(0.03)  # smallest: *3
    assert adjusted[2] == pytest.approx(0.06)  # second: *2
    assert adjusted[1] == pytest.approx(0.06)  # third: *1 -> 0.04, but monotone -> 0.06
    assert np.isnan(adjusted[3])
