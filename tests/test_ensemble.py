from datetime import date, timedelta

import numpy as np
import pytest

from w8t.core import metrics
from w8t.forecasting.base import InsufficientDataError
from w8t.forecasting.baselines import LinearTrend, NaiveLastValue
from w8t.forecasting.ensemble import EqualWeightEnsemble, _mixture_quantile

D0 = date(2026, 1, 1)


def series_from(values):
    return metrics.to_series((D0 + timedelta(days=d), v) for d, v in enumerate(values))


@pytest.fixture
def trending():
    rng = np.random.default_rng(0)
    return series_from(85 - 0.1 * np.arange(40) + rng.normal(0, 0.3, 40))


def test_mean_is_average_of_members(trending):
    ens = EqualWeightEnsemble([NaiveLastValue(), LinearTrend()], name="e").fit(trending)
    a = NaiveLastValue().fit(trending).predict([1, 7, 30])
    b = LinearTrend().fit(trending).predict([1, 7, 30])

    np.testing.assert_allclose(ens.predict([1, 7, 30]).mean, (a.mean + b.mean) / 2)


def test_identical_members_reproduce_member_interval(trending):
    single = LinearTrend().fit(trending).predict([1, 7, 30])
    ens = EqualWeightEnsemble([LinearTrend(), LinearTrend()], name="e").fit(trending)
    fc = ens.predict([1, 7, 30])

    # LinearTrend uses a t-quantile; the ensemble re-reads it as Gaussian with the same width,
    # so the mixture of two identical Gaussians gives back the same bounds.
    np.testing.assert_allclose(fc.lower, single.lower, atol=1e-4)
    np.testing.assert_allclose(fc.upper, single.upper, atol=1e-4)


def test_disagreement_widens_mixture_beyond_quantile_average(trending):
    members = [NaiveLastValue(), LinearTrend()]  # disagree at long horizons on a trend
    mix = EqualWeightEnsemble(members, interval="mixture", name="m").fit(trending)
    avg = EqualWeightEnsemble(members, interval="quantile_avg", name="a").fit(trending)
    fm, fa = mix.predict([30]), avg.predict([30])

    assert fm.mean[0] == pytest.approx(fa.mean[0])
    assert (fm.upper[0] - fm.lower[0]) >= (fa.upper[0] - fa.lower[0]) - 1e-9


def test_mixture_quantile_matches_monte_carlo():
    rng = np.random.default_rng(1)
    means, sds = np.array([80.0, 82.0]), np.array([0.5, 1.0])
    draws = np.concatenate([rng.normal(80, 0.5, 200_000), rng.normal(82, 1.0, 200_000)])

    for q in (0.025, 0.5, 0.975):
        assert _mixture_quantile(q, means, sds) == pytest.approx(np.quantile(draws, q), abs=0.02)


def test_gate_requires_every_member():
    # 10 points overall (passes the ensemble's own min_obs), but only 5 inside LinearTrend's
    # 28-day window -> that member refuses, so the whole ensemble must refuse too.
    days = [0, 1, 2, 3, 4, 50, 51, 52, 53, 54]
    s = metrics.to_series((D0 + timedelta(days=d), 80.0 + 0.1 * i) for i, d in enumerate(days))
    NaiveLastValue().fit(s)  # this member alone would be fine
    ens = EqualWeightEnsemble([NaiveLastValue(), LinearTrend()], name="e")

    with pytest.raises(InsufficientDataError, match="Regressão linear"):
        ens.fit(s)


def test_prototypes_are_not_mutated(trending):
    naive = NaiveLastValue()
    EqualWeightEnsemble([naive, LinearTrend()], name="e").fit(trending)

    assert naive._origin is None  # still unfitted


def test_unknown_interval_rejected():
    with pytest.raises(ValueError):
        EqualWeightEnsemble([NaiveLastValue()], interval="x", name="e")


def test_backtest_reuse_of_member_fits_is_equivalent_to_refitting(trending):
    from w8t.forecasting.backtest import walk_forward

    def ens():
        return EqualWeightEnsemble([NaiveLastValue(), LinearTrend()], name="e")

    alone = walk_forward(trending, [ens()], horizons=[1, 7], step_days=3, min_history_days=10)
    reused = walk_forward(
        trending, [NaiveLastValue(), LinearTrend(), ens()],
        horizons=[1, 7], step_days=3, min_history_days=10,
    )
    a = alone.forecasts.reset_index(drop=True)
    r = reused.forecasts[reused.forecasts["model"] == "e"].reset_index(drop=True)
    assert len(a) > 0
    np.testing.assert_allclose(r[["mean", "lower", "upper"]], a[["mean", "lower", "upper"]])


def test_fit_from_fitted_rejects_mismatched_members(trending):
    ens = EqualWeightEnsemble([NaiveLastValue(), LinearTrend()], name="e")
    with pytest.raises(ValueError):
        ens.fit_from_fitted([LinearTrend().fit(trending), NaiveLastValue().fit(trending)])
    with pytest.raises(ValueError):
        ens.fit_from_fitted([NaiveLastValue().fit(trending), LinearTrend().fit(trending[:-3])])
