"""Multi-series benchmark: compare forecasters across many independent synthetic series.

Why: a single ~6-month series can't separate the models (see ``significance``: at 14-30 days it
holds too few independent windows to test anything), and choosing kernels/windows by looking at
the demo series alone is selection bias. Independent series are independent evidence, so here
every model is backtested on several scenarios x seeds and compared with a *paired test across
series* (Wilcoxon signed-rank on per-series MAE differences) - valid because series don't share
noise.

Scenarios deliberately cover shapes the demo doesn't: gain phases, pure maintenance with a
weekly pattern, sparse logging with long gaps, a cut-to-bulk regime change, noisy scales.
All synthetic - never mixed with real data.

Run: ``python -m w8t.forecasting.benchmark`` (a few minutes; writes docs/benchmark.md).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from w8t.core import metrics
from w8t.forecasting.backtest import DEFAULT_HORIZONS, walk_forward
from w8t.forecasting.base import ForecastModel
from w8t.forecasting.significance import _holm

N_DAYS = 180
START = date(2026, 1, 1)

Scenario = Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]]


def _observe(truth: np.ndarray, rng: np.random.Generator, sd: float, missing: float):
    keep = rng.random(len(truth)) >= missing
    keep[0] = keep[-1] = True
    days = np.flatnonzero(keep)
    return days, np.round(truth[days] + rng.normal(0, sd, len(days)), 1)


def cutting_plateau(rng):
    d = np.arange(N_DAYS)
    trend = 88 - 0.08 * np.minimum(d, 90) + 0.01 * np.clip(d - 120, 0, None)
    return _observe(trend, rng, 0.35, 0.15)


def bulk(rng):
    trend = 70 + 0.04 * np.arange(N_DAYS)
    return _observe(trend, rng, 0.4, 0.15)


def maintenance_weekly(rng):
    d = np.arange(N_DAYS)
    weekly = np.where(d % 7 < 2, 0.5, 0.0)  # e.g. heavier after the weekend
    return _observe(75 + weekly, rng, 0.3, 0.15)


def sparse_long_gaps(rng):
    trend = 95 - 0.06 * np.arange(N_DAYS)
    days, y = _observe(trend, rng, 0.35, 0.4)
    gap_starts = rng.choice(np.arange(30, N_DAYS - 45), size=2, replace=False)
    in_gap = np.zeros_like(days, dtype=bool)
    for g in gap_starts:
        in_gap |= (days >= g) & (days < g + 12)
    return days[~in_gap], y[~in_gap]


def cut_then_bulk(rng):
    d = np.arange(N_DAYS)
    trend = np.where(d < 90, 85 - 0.1 * d, 76 + 0.05 * (d - 90))
    return _observe(trend, rng, 0.35, 0.15)


def noisy_slow_loss(rng):
    trend = 90 - 0.03 * np.arange(N_DAYS)
    return _observe(trend, rng, 0.7, 0.15)


SCENARIOS: dict[str, Scenario] = {
    "cutting→platô": cutting_plateau,
    "bulk": bulk,
    "manutenção semanal": maintenance_weekly,
    "esparso c/ gaps longos": sparse_long_gaps,
    "cutting→bulk": cut_then_bulk,
    "perda lenta ruidosa": noisy_slow_loss,
}


def scenario_series(name: str, seed: int) -> pd.Series:
    days, y = SCENARIOS[name](np.random.default_rng(seed))
    return metrics.to_series(
        (START + timedelta(days=int(d)), float(v)) for d, v in zip(days, y, strict=True)
    )


def run_benchmark(
    models: Sequence[ForecastModel],
    *,
    scenarios: Sequence[str] | None = None,
    seeds: Sequence[int] = range(5),
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    step_days: int = 3,
) -> pd.DataFrame:
    """One row per (scenario, seed, model, horizon) with MAE and coverage on common cases."""
    rows = []
    for name in scenarios or SCENARIOS:
        for seed in seeds:
            result = walk_forward(
                scenario_series(name, seed), models, horizons, step_days=step_days
            )
            for r in result.summary().itertuples():
                rows.append(
                    {
                        "scenario": name, "seed": seed, "model": r.model, "horizon": r.horizon,
                        "n": r.n, "mae": r.mae, "coverage": r.coverage,
                        "mean_width": r.mean_width, "bias": r.bias,
                    }
                )
    return pd.DataFrame(rows)


def overall(per_series: pd.DataFrame) -> pd.DataFrame:
    """Per (model, horizon): mean MAE, mean coverage, mean rank of MAE within each series."""
    ranked = per_series.assign(
        rank=per_series.groupby(["scenario", "seed", "horizon"])["mae"].rank()
    )
    return (
        ranked.groupby(["horizon", "model"])
        .agg(mae=("mae", "mean"), coverage=("coverage", "mean"), mean_rank=("rank", "mean"),
             series=("mae", "size"))
        .reset_index()
        .sort_values(["horizon", "mae"])
    )


def paired_vs_best(per_series: pd.DataFrame) -> pd.DataFrame:
    """Per horizon: best mean-MAE model vs. each other, Wilcoxon signed-rank on per-series MAE
    differences (series are independent), Holm-adjusted within the horizon."""
    rows = []
    for horizon, group in per_series.groupby("horizon"):
        wide = group.pivot_table(index=["scenario", "seed"], columns="model", values="mae")
        wide = wide.dropna()
        best = wide.mean().idxmin()
        pairs = []
        for other in wide.columns.drop(best):
            diff = wide[best] - wide[other]
            p = stats.wilcoxon(diff).pvalue if (diff != 0).any() else float("nan")
            pairs.append((other, float(diff.mean()), float((diff < 0).mean()), p))
        holm = _holm([p for *_, p in pairs])
        rows += [
            {
                "horizon": int(horizon), "best": best, "other": other, "series": len(wide),
                "mean_mae_diff": d, "best_wins_share": w, "p_value": p, "p_holm": ph,
            }
            for (other, d, w, p), ph in zip(pairs, holm, strict=True)
        ]
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def to_markdown(per_series: pd.DataFrame, step_days: int, seeds: int) -> str:
    ov = overall(per_series)
    by_scenario = (
        per_series.groupby(["scenario", "horizon", "model"])[["mae", "coverage"]]
        .mean()
        .reset_index()
    )
    tests = paired_vs_best(per_series)
    fmt = {"mae": "{:.3f}", "coverage": "{:.0%}", "mean_rank": "{:.2f}"}

    def table(df, cols, formats):
        out = df[cols].copy()
        for c, f in formats.items():
            if c in out:
                out[c] = out[c].map(lambda v, f=f: "—" if pd.isna(v) else f.format(v))
        return _markdown_table(out)

    parts = [
        "# Benchmark multi-série de previsão",
        "",
        (
            f"Gerado por `python -m w8t.forecasting.benchmark`. {len(SCENARIOS)} cenários "
            f"sintéticos × {seeds} sementes = "
            f"{per_series.groupby(['scenario', 'seed']).ngroups} séries independentes de "
            f"{N_DAYS} dias; walk-forward a cada {step_days} dias; métricas nos casos comuns a "
            "todos os modelos em cada série. Cobertura = fração dos valores reais dentro do "
            "intervalo de 95%."
        ),
        "",
        "## Geral (média entre séries)",
        "",
        table(ov, ["horizon", "model", "mae", "coverage", "mean_rank", "series"], fmt),
        "",
        "## Melhor modelo vs. demais (Wilcoxon pareado entre séries, Holm por horizonte)",
        "",
        table(
            tests,
            ["horizon", "best", "other", "series", "mean_mae_diff", "best_wins_share",
             "p_value", "p_holm"],
            {"mean_mae_diff": "{:+.3f}", "best_wins_share": "{:.0%}", "p_value": "{:.4f}",
             "p_holm": "{:.4f}"},
        ),
        "",
        "## Por cenário",
        "",
    ]
    for name in SCENARIOS:
        sub = by_scenario[by_scenario["scenario"] == name].sort_values(["horizon", "mae"])
        parts += [f"### {name}", "", table(sub, ["horizon", "model", "mae", "coverage"], fmt), ""]
    return "\n".join(parts)


def main() -> None:  # pragma: no cover - CLI
    from w8t.forecasting.registry import all_models

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--step-days", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("docs/benchmark.md"))
    args = parser.parse_args()

    per_series = run_benchmark(all_models(), seeds=range(args.seeds), step_days=args.step_days)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(to_markdown(per_series, args.step_days, args.seeds), encoding="utf-8")
    per_series.to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"Escrito {args.out} e {args.out.with_suffix('.csv')}")


if __name__ == "__main__":  # pragma: no cover
    main()
