from datetime import date, timedelta
from itertools import pairwise
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from w8t.core import metrics
from w8t.forecasting.backtest import forecast_origins, summarize, walk_forward
from w8t.forecasting.base import ForecastModel, InsufficientDataError
from w8t.forecasting.baselines import LinearTrend, NaiveLastValue, baseline_models

D0 = date(2026, 1, 1)


def series_from(values, days=None):
    days = range(len(values)) if days is None else days
    return metrics.to_series((D0 + timedelta(days=d), v) for d, v in zip(days, values, strict=True))


class SpyModel(ForecastModel):
    """Records the last date it was allowed to see; predicts that date's value, fixed ±1."""

    name = "spy"
    min_obs = 1
    seen_until: ClassVar[list] = []  # class-level so deep copies report back

    def _fit(self, series):
        SpyModel.seen_until.append(series.index[-1])
        self._last = float(series.iloc[-1])

    def _predict(self, horizons, level):
        mean = np.full(len(horizons), self._last)
        return mean, mean - 1, mean + 1


class PickyModel(SpyModel):
    """Refuses to fit on short histories - exercises gating/common-case logic."""

    name = "picky"

    def _fit(self, series):
        if len(series) < 50:
            raise InsufficientDataError("picky: poucos dados")
        super()._fit(series)


@pytest.fixture(autouse=True)
def _reset_spy():
    SpyModel.seen_until = []


def test_origins_respect_warmup_and_step():
    s = series_from([80.0] * 60)
    origins = forecast_origins(s, step_days=7, min_history_days=28)

    assert origins[0] == pd.Timestamp(D0 + timedelta(days=28))
    assert all((b - a).days == 7 for a, b in pairwise(origins))


def test_origins_with_gaps_use_next_real_measurement():
    s = series_from([80.0] * 5, days=[0, 10, 29, 33, 40])
    origins = forecast_origins(s, step_days=7, min_history_days=28)

    # day 29 is the first date past warmup; day 33 is < 7 days later; day 40 qualifies.
    assert origins == [pd.Timestamp(D0 + timedelta(days=d)) for d in (29, 40)]


def test_model_never_sees_data_after_origin():
    s = series_from(np.arange(60, dtype=float))
    result = walk_forward(s, [SpyModel()], horizons=[1, 7], step_days=5, min_history_days=10)

    origins = result.forecasts["origin"].unique()
    assert list(SpyModel.seen_until) == list(origins)
    # Spy predicts last-seen value; truth rises 1/day -> error equals the horizon exactly.
    assert (result.forecasts["error"] == result.forecasts["horizon"]).all()


def test_targets_without_real_measurement_are_not_scored():
    days = [d for d in range(60) if d != 36]  # day 36 missing
    s = series_from([80.0] * len(days), days=days)
    result = walk_forward(s, [SpyModel()], horizons=[1, 7], step_days=100, min_history_days=29)

    # single origin at day 29: h=1 -> day 30 (exists), h=7 -> day 36 (missing, not scored)
    assert list(result.forecasts["horizon"]) == [1]


def test_metrics_on_known_errors():
    fc = pd.DataFrame(
        {
            "model": ["a"] * 4,
            "origin": pd.to_datetime(["2026-01-01", "2026-01-08", "2026-01-15", "2026-01-22"]),
            "horizon": [1] * 4,
            "error": [1.0, -1.0, 3.0, -3.0],
            "covered": [True, True, False, True],
            "width": [2.0, 2.0, 2.0, 4.0],
        }
    )
    row = summarize(fc).iloc[0]

    assert row["n"] == 4
    assert row["mae"] == pytest.approx(2.0)
    assert row["rmse"] == pytest.approx(np.sqrt(5.0))
    assert row["bias"] == pytest.approx(0.0)
    assert row["coverage"] == pytest.approx(0.75)
    assert row["mean_width"] == pytest.approx(2.5)


def test_gated_model_is_skipped_and_comparison_uses_common_cases():
    s = series_from(np.arange(80, dtype=float))
    result = walk_forward(
        s, [SpyModel(), PickyModel()], horizons=[1], step_days=5, min_history_days=10
    )

    assert set(result.skipped["model"]) == {"picky"}
    assert "poucos dados" in result.skipped["reason"].iloc[0]

    common = result.summary()
    everything = result.summary(common_only=False)
    n_common = common.set_index("model")["n"]
    assert n_common["spy"] == n_common["picky"]
    assert everything.set_index("model")["n"]["spy"] > n_common["spy"]


def test_skill_vs_reference():
    rng = np.random.default_rng(0)
    s = series_from(90 - 0.1 * np.arange(120) + rng.normal(0, 0.2, 120))
    summary = walk_forward(s, [NaiveLastValue(), LinearTrend(28)], horizons=[14]).summary(
        reference="Último valor"
    )
    skill = summary.set_index("model")["skill_vs_ref"]

    assert skill["Último valor"] == pytest.approx(0.0)
    assert skill["Regressão linear 28d"] > 0.5  # clean linear trend: regression must win


def test_input_series_is_not_modified():
    rng = np.random.default_rng(1)
    s = series_from(80 + rng.normal(0, 0.3, 60))
    original = s.copy()
    walk_forward(s, baseline_models(), horizons=[1, 7])

    assert s.equals(original)
