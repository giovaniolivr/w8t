"""Labeled evaluation: baseline detectors (``w8t.core``) vs. Kalman detectors (``w8t.patterns``).

Synthetic series where the truth is known: the true slope on every day and the days where
anomalies were planted. Every series is used for all three tasks (so anomalies are present while
detecting trend/plateau, as in real data):

- Trend: at checkpoints every 7 days (from day 35), each detector sees only data up to the
  checkpoint and classifies the current direction. Truth from the true slope at the checkpoint
  (down <= -band, up >= band, else stable, band = 0.25 kg/week). Reported: correct, *wrong*
  (a decided verdict that disagrees with the truth - the costly error) and undetermined rates.
- Plateau: day-level labels on measured days from day 21 on: truly flat = |true slope| < band.
  A day is detected flat if it falls inside a reported plateau. Precision / recall / F1.
- Anomaly: planted spikes of 1.5-2.5 kg (either sign). Recall = planted ones flagged; false
  positives per 100 measurements = flagged points that weren't planted.

Run: ``python -m w8t.patterns.evaluation`` (writes docs/patterns_benchmark.md and .csv).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from w8t.core import anomaly as base_anomaly
from w8t.core import metrics
from w8t.core import plateau as base_plateau
from w8t.core import trend as base_trend
from w8t.core.trend import STABLE_BAND_KG_PER_WEEK, TrendDirection
from w8t.patterns import kalman as kal

N_DAYS = 180
START = date(2026, 1, 1)
NOISE_SD = 0.35
MISSING = 0.15
N_ANOMALIES = 3
TREND_FIRST_CHECKPOINT = 35
TREND_EVERY = 7
PLATEAU_WARMUP = 21


@dataclass(frozen=True)
class LabeledSeries:
    scenario: str
    seed: int
    series: pd.Series
    true_slope_per_day: np.ndarray  # length N_DAYS, index = day offset
    anomaly_dates: frozenset[date]


def _piecewise(breaks: list[tuple[int, float]]) -> np.ndarray:
    """Slope per day from (start_day, slope) breakpoints."""
    slope = np.zeros(N_DAYS)
    for (start, value), (end, _) in zip(breaks, [*breaks[1:], (N_DAYS, 0.0)], strict=True):
        slope[start:end] = value
    return slope


SCENARIOS: dict[str, Callable[[], tuple[float, np.ndarray, np.ndarray]]] = {
    # name -> (start weight, slope per day, extra deterministic pattern)
    "cutting→platô": lambda: (88.0, _piecewise([(0, -0.08), (90, 0.0), (120, 0.01)]), None),
    "bulk": lambda: (70.0, _piecewise([(0, 0.04)]), None),
    "manutenção semanal": lambda: (
        75.0, _piecewise([(0, 0.0)]), np.where(np.arange(N_DAYS) % 7 < 2, 0.5, 0.0)
    ),
    "cutting→bulk": lambda: (85.0, _piecewise([(0, -0.1), (90, 0.05)]), None),
    "cutting rápido": lambda: (95.0, _piecewise([(0, -0.12)]), None),
    "manutenção→cutting": lambda: (80.0, _piecewise([(0, 0.0), (80, -0.08)]), None),
}


def labeled_series(scenario: str, seed: int) -> LabeledSeries:
    rng = np.random.default_rng(seed)
    start_kg, slope, pattern = SCENARIOS[scenario]()
    truth = start_kg + np.concatenate([[0.0], np.cumsum(slope[:-1])])
    if pattern is not None:
        truth = truth + pattern
    keep = rng.random(N_DAYS) >= MISSING
    keep[0] = keep[-1] = True
    days = np.flatnonzero(keep)
    y = truth[days] + rng.normal(0, NOISE_SD, len(days))

    eligible = days[days >= 30]
    planted: list[int] = []
    while len(planted) < N_ANOMALIES:
        d = int(rng.choice(eligible))
        if all(abs(d - p) >= 10 for p in planted):
            planted.append(d)
    for d in planted:
        y[np.searchsorted(days, d)] += rng.choice([-1, 1]) * rng.uniform(1.5, 2.5)

    series = metrics.to_series(
        (START + timedelta(days=int(d)), round(float(v), 1)) for d, v in zip(days, y, strict=True)
    )
    return LabeledSeries(
        scenario, seed, series, slope, frozenset(START + timedelta(days=d) for d in planted)
    )


def _truth_direction(slope_per_day: float, band: float) -> TrendDirection:
    weekly = slope_per_day * 7
    if weekly <= -band:
        return TrendDirection.DOWN
    if weekly >= band:
        return TrendDirection.UP
    return TrendDirection.STABLE


DETECTORS = {
    "baseline": {
        "trend": base_trend.current_trend,
        "plateau": base_plateau.detect_plateaus,
        "anomaly": base_anomaly.detect_anomalies,
    },
    "kalman": {
        "trend": kal.current_trend,
        "plateau": kal.detect_plateaus,
        "anomaly": kal.detect_anomalies,
    },
}


def evaluate_series(ls: LabeledSeries, band: float = STABLE_BAND_KG_PER_WEEK) -> list[dict]:
    rows = []
    s = ls.series
    offsets = (s.index - pd.Timestamp(START)).days.to_numpy()
    for method, det in DETECTORS.items():
        row = {"scenario": ls.scenario, "seed": ls.seed, "method": method}

        # --- trend (past-only at each checkpoint)
        correct = wrong = undetermined = 0
        for cp in range(TREND_FIRST_CHECKPOINT, N_DAYS, TREND_EVERY):
            past = s[offsets <= cp]
            verdict = det["trend"](past)
            truth = _truth_direction(ls.true_slope_per_day[cp], band)
            if verdict is None or verdict.direction is TrendDirection.UNDETERMINED:
                undetermined += 1
            elif verdict.direction is truth:
                correct += 1
            else:
                wrong += 1
        row |= {"trend_correct": correct, "trend_wrong": wrong, "trend_undetermined": undetermined}

        # --- plateau (day level, measured days after warmup)
        found = det["plateau"](s)
        detected = np.zeros(len(s), dtype=bool)
        for p in found:
            detected |= (s.index.date >= p.start) & (s.index.date <= p.end)
        truly_flat = np.abs(ls.true_slope_per_day[offsets] * 7) < band
        scored = offsets >= PLATEAU_WARMUP
        row |= {
            "plateau_tp": int((detected & truly_flat & scored).sum()),
            "plateau_fp": int((detected & ~truly_flat & scored).sum()),
            "plateau_fn": int((~detected & truly_flat & scored).sum()),
        }

        # --- anomaly
        flags = det["anomaly"](s)
        flagged = set(flags.index[flags["is_anomaly"]].date)
        row |= {
            "anomaly_planted": len(ls.anomaly_dates),
            "anomaly_tp": len(flagged & ls.anomaly_dates),
            "anomaly_fp": len(flagged - ls.anomaly_dates),
            "anomaly_scored": int(flags["robust_z"].notna().sum()),
        }
        rows.append(row)
    return rows


def run(seeds=range(8), scenarios=None) -> pd.DataFrame:
    rows = []
    for name in scenarios or SCENARIOS:
        for seed in seeds:
            rows += evaluate_series(labeled_series(name, seed))
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = results.groupby(by).sum(numeric_only=True)
    checkpoints = g["trend_correct"] + g["trend_wrong"] + g["trend_undetermined"]
    precision = g["plateau_tp"] / (g["plateau_tp"] + g["plateau_fp"])
    recall = g["plateau_tp"] / (g["plateau_tp"] + g["plateau_fn"])
    return pd.DataFrame(
        {
            "trend_correct": g["trend_correct"] / checkpoints,
            "trend_wrong": g["trend_wrong"] / checkpoints,
            "trend_undetermined": g["trend_undetermined"] / checkpoints,
            "plateau_precision": precision,
            "plateau_recall": recall,
            "plateau_f1": 2 * precision * recall / (precision + recall),
            "anomaly_recall": g["anomaly_tp"] / g["anomaly_planted"],
            "anomaly_fp_per_100": 100 * g["anomaly_fp"] / g["anomaly_scored"],
        }
    ).reset_index()


def _markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def to_markdown(results: pd.DataFrame, seeds: int) -> str:
    def fmt(df):
        out = df.copy()
        for c in out.columns:
            if c.startswith(("trend_", "plateau_", "anomaly_recall")):
                out[c] = out[c].map(lambda v: "—" if pd.isna(v) else f"{v:.0%}")
            elif c == "anomaly_fp_per_100":
                out[c] = out[c].map(lambda v: f"{v:.2f}")
        return _markdown_table(out)

    n_series = results.groupby(["scenario", "seed"]).ngroups
    return "\n".join(
        [
            "# Avaliação rotulada: detectores baseline vs. Kalman",
            "",
            (
                f"Gerado por `python -m w8t.patterns.evaluation`. {len(SCENARIOS)} cenários × "
                f"{seeds} sementes = {n_series} séries sintéticas de {N_DAYS} dias (ruído "
                f"{NOISE_SD} kg, {MISSING:.0%} de dias faltando, {N_ANOMALIES} anomalias de "
                "1,5-2,5 kg plantadas por série). Tendência avaliada só com dados até cada ponto "
                f"de checagem (a cada {TREND_EVERY} dias); platô por dia medido; anomalia por "
                "ponto plantado. Faixa de estabilidade: ±0,25 kg/semana."
            ),
            "",
            "## Geral",
            "",
            fmt(summarize(results, ["method"])),
            "",
            "## Por cenário",
            "",
            fmt(summarize(results, ["scenario", "method"])),
            "",
        ]
    )


def main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("docs/patterns_benchmark.md"))
    args = parser.parse_args()
    results = run(seeds=range(args.seeds))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(to_markdown(results, args.seeds), encoding="utf-8")
    results.to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"Escrito {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
