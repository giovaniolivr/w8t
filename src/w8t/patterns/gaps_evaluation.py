"""Labeled evaluation of gap reconstruction methods.

On the same 48 labeled synthetic series used for the detectors (``w8t.patterns.evaluation``),
real measurements are deleted over artificial holes of 3, 7 and 14 days (two holes per length per
series, away from the edges), each method reconstructs the resulting gap, and the estimates are
compared with:

- the **held-out measurements** (what the scale actually read): MAE and coverage of the 95%
  interval - the interval is for a measurement, so ~95% is the target;
- the **true noise-free level** (known in synthetic data): MAE - how well the method recovers the
  underlying weight rather than chasing noise.

Planted anomalies are excluded from the scored held-out points (they're not "what the scale would
typically read"), but stay in the data the methods see, as they would in real life.

Run: ``python -m w8t.patterns.gaps_evaluation`` (writes docs/gaps_benchmark.md and .csv).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from w8t.patterns import evaluation as ev
from w8t.patterns.gaps import METHODS, CannotReconstructError, find_gaps, reconstruct

HOLE_LENGTHS = (3, 7, 14)
HOLES_PER_LENGTH = 2
EDGE_MARGIN = 30


def evaluate_series(ls: ev.LabeledSeries) -> list[dict]:
    rng = np.random.default_rng(10_000 + ls.seed)
    s = ls.series
    offsets = (s.index - pd.Timestamp(ev.START)).days.to_numpy()
    rows = []
    for length in HOLE_LENGTHS:
        for _ in range(HOLES_PER_LENGTH):
            hole_start = int(rng.integers(EDGE_MARGIN, ev.N_DAYS - EDGE_MARGIN - length))
            in_hole = (offsets >= hole_start) & (offsets < hole_start + length)
            held = s[in_hole]
            held = held[~np.isin(held.index.date, list(ls.anomaly_dates))]
            if held.empty:
                continue
            cut = s[~in_hole]
            gap = next(
                g for g in find_gaps(cut)
                if g.start <= held.index[0].date() <= g.end
            )
            for method in METHODS:
                base = {
                    "scenario": ls.scenario, "seed": ls.seed, "hole_days": length,
                    "gap_days": gap.days, "method": method,
                }
                try:
                    rec = reconstruct(cut, gap, method).loc[held.index]
                except CannotReconstructError:
                    rows.append(base | {"failed": True})
                    continue
                truth = ls.true_level[(held.index - pd.Timestamp(ev.START)).days]
                for (ts, real), est, lo, hi, level in zip(
                    held.items(), rec["estimate_kg"], rec["lower"], rec["upper"], truth,
                    strict=True,
                ):
                    rows.append(
                        base
                        | {
                            "failed": False, "date": ts.date(),
                            "abs_err_meas": abs(real - est), "covered": lo <= real <= hi,
                            "width": hi - lo, "abs_err_level": abs(level - est),
                        }
                    )
    return rows


def run(seeds=range(8), scenarios=None) -> pd.DataFrame:
    rows = []
    for name in scenarios or ev.SCENARIOS:
        for seed in seeds:
            rows += evaluate_series(ev.labeled_series(name, seed))
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    ok = results[~results["failed"]]
    out = ok.groupby(by).agg(
        points=("abs_err_meas", "size"),
        mae_vs_measurement=("abs_err_meas", "mean"),
        coverage=("covered", "mean"),
        mean_width=("width", "mean"),
        mae_vs_true_level=("abs_err_level", "mean"),
    )
    failures = results[results["failed"]].groupby(by).size()
    out["failed_gaps"] = failures.reindex(out.index, fill_value=0)
    return out.reset_index()


def to_markdown(results: pd.DataFrame, seeds: int) -> str:
    def fmt(df):
        out = df.copy()
        for c in ("mae_vs_measurement", "mean_width", "mae_vs_true_level"):
            out[c] = out[c].map(lambda v: f"{v:.3f}")
        out["coverage"] = out["coverage"].map(lambda v: f"{v:.0%}")
        return ev._markdown_table(out)

    n_series = results.groupby(["scenario", "seed"]).ngroups
    return "\n".join(
        [
            "# Avaliação rotulada: reconstrução de lacunas",
            "",
            (
                f"Gerado por `python -m w8t.patterns.gaps_evaluation`. {n_series} séries "
                f"sintéticas ({len(ev.SCENARIOS)} cenários × {seeds} sementes); em cada uma, "
                f"{HOLES_PER_LENGTH} buracos artificiais de {', '.join(map(str, HOLE_LENGTHS))} "
                "dias (medições reais apagadas). Cobertura = fração das medições apagadas dentro "
                "do intervalo de 95% (alvo ~95%). *mae_vs_true_level* compara com o peso "
                "verdadeiro sem ruído. Anomalias plantadas não são pontuadas."
            ),
            "",
            "## Por método",
            "",
            fmt(summarize(results, ["method"])),
            "",
            "## Por tamanho do buraco",
            "",
            fmt(summarize(results, ["hole_days", "method"])),
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
    parser.add_argument("--out", type=Path, default=Path("docs/gaps_benchmark.md"))
    args = parser.parse_args()
    results = run(seeds=range(args.seeds))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(to_markdown(results, args.seeds), encoding="utf-8")
    results.to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"Escrito {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
