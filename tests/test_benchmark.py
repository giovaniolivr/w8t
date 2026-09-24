import numpy as np
import pandas as pd
import pytest

from w8t.forecasting import benchmark
from w8t.forecasting.baselines import LinearTrend, NaiveLastValue


@pytest.mark.parametrize("name", list(benchmark.SCENARIOS))
def test_scenarios_are_deterministic_ordered_and_realistic(name):
    a = benchmark.scenario_series(name, seed=0)
    b = benchmark.scenario_series(name, seed=0)
    c = benchmark.scenario_series(name, seed=1)

    assert a.equals(b)
    assert not a.equals(c)
    assert a.index.is_monotonic_increasing and not a.index.has_duplicates
    assert (a.index[-1] - a.index[0]).days == benchmark.N_DAYS - 1
    assert a.between(40, 200).all()
    # weights recorded with 0.1 kg resolution, like real entries
    np.testing.assert_allclose(a * 10, np.round(a * 10))


def test_sparse_scenario_has_long_gaps():
    s = benchmark.scenario_series("esparso c/ gaps longos", seed=0)
    assert s.index.to_series().diff().dt.days.max() >= 12


def test_run_benchmark_and_reports():
    per_series = benchmark.run_benchmark(
        [NaiveLastValue(), LinearTrend()],
        scenarios=["bulk", "cutting→platô"],
        seeds=range(3),
        horizons=[1, 7],
        step_days=7,
    )

    assert set(per_series["model"]) == {"Último valor", "Regressão linear 28d"}
    assert len(per_series) == 2 * 3 * 2 * 2  # scenarios x seeds x models x horizons
    ov = benchmark.overall(per_series)
    assert (ov["series"] == 6).all()
    # 2 models: ranks 1 and 2 in every series -> mean ranks per horizon sum to 3
    np.testing.assert_allclose(ov.groupby("horizon")["mean_rank"].sum(), 3.0)

    tests = benchmark.paired_vs_best(per_series)
    assert (tests["series"] == 6).all()
    assert tests["p_holm"].ge(tests["p_value"]).all()

    md = benchmark.to_markdown(per_series, step_days=7, seeds=3)
    assert md.startswith("# Benchmark multi-série")
    assert "| horizon | model |" in md


def test_paired_test_detects_consistent_winner():
    rows = [
        {"scenario": "s", "seed": i, "model": m, "horizon": 7, "mae": mae}
        for i in range(12)
        for m, mae in (("good", 0.3 + 0.01 * i), ("bad", 0.5 + 0.01 * i))
    ]
    tests = benchmark.paired_vs_best(pd.DataFrame(rows))

    row = tests.iloc[0]
    assert row["best"] == "good" and row["other"] == "bad"
    assert row["best_wins_share"] == 1.0
    assert row["p_value"] < 0.001
