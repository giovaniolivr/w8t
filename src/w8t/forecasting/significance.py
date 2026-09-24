"""Is one model's forecast error *really* lower than another's? Diebold-Mariano test.

Given two models' errors on the same (origin, horizon) cases, the loss differential
``d_t = L(e_a,t) - L(e_b,t)`` is tested for zero mean. Forecasts whose horizon exceeds the spacing
between origins overlap in time, so consecutive ``d_t`` are autocorrelated and a plain paired
t-test would overstate significance (measured: >25% false rejections at nominal 5% for h=7,
daily origins). The long-run variance of ``d`` includes autocovariances up to lag
``L = ceil(h / step) - 1`` (the overlap) with the original DM uniform weights, plus the
Harvey-Leybourne-Newbold (1997) small-sample correction, compared to Student t with n-1 df.

Honest limits (simulated with MA(L) differentials, 2026-09-24):
- Even so the test is liberal: ~6-10% false rejections at nominal 5% (uniform weights; Bartlett
  weights were worse, ~12-14%). Read p < 0.05 as weak evidence, p < 0.01 as solid.
- With few *effectively independent* cases (``n / (L + 1)``) it breaks down entirely, so below
  ``MIN_EFFECTIVE_CASES`` no test is run and the comparison is reported as not testable - e.g.
  ~6 months of data at h=30 holds only ~3 independent 30-day windows; nothing can be claimed.

Negative mean differential = model A has lower loss. Two-sided p-value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

MIN_CASES = 10
MIN_EFFECTIVE_CASES = 10


@dataclass(frozen=True)
class DMResult:
    model_a: str
    model_b: str
    horizon: int
    n: int
    mean_loss_diff: float  # mean(L_a - L_b); < 0 means A better
    statistic: float
    p_value: float
    lags: int

    @property
    def effective_cases(self) -> float:
        return self.n / (self.lags + 1)


def dm_test(d: np.ndarray, lags: int) -> tuple[float, float]:
    """(HLN-corrected DM statistic, two-sided p-value) for loss differentials ``d``."""
    n = len(d)
    centered = d - d.mean()
    long_run = centered @ centered / n
    for k in range(1, lags + 1):
        long_run += 2 * (centered[k:] @ centered[:-k] / n)
    if long_run <= 0:
        # identical losses, or (rare with uniform weights) a negative variance estimate
        return float("nan"), float("nan")
    dm = d.mean() / math.sqrt(long_run / n)
    h = lags + 1
    statistic = dm * math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(statistic), float(2 * stats.t.sf(abs(statistic), df=n - 1))


def compare(
    forecasts: pd.DataFrame,
    model_a: str,
    model_b: str,
    horizon: int,
    *,
    step_days: int,
    loss: str = "abs",
) -> DMResult | None:
    """DM test on the cases both models forecast at ``horizon``.

    ``None`` if there are too few cases, or too few *effectively independent* ones.
    """
    sub = forecasts[(forecasts["horizon"] == horizon) & forecasts["model"].isin([model_a, model_b])]
    wide = sub.pivot_table(index="origin", columns="model", values="error").dropna()
    if len(wide) < MIN_CASES or model_a not in wide or model_b not in wide:
        return None
    wide = wide.sort_index()
    loss_fn = np.abs if loss == "abs" else np.square
    d = (loss_fn(wide[model_a]) - loss_fn(wide[model_b])).to_numpy()
    lags = max(math.ceil(horizon / step_days) - 1, 0)
    if len(d) / (lags + 1) < MIN_EFFECTIVE_CASES:
        return None
    statistic, p_value = dm_test(d, lags)
    return DMResult(model_a, model_b, horizon, len(d), float(d.mean()), statistic, p_value, lags)


def versus_best(forecasts: pd.DataFrame, *, step_days: int, loss: str = "abs") -> pd.DataFrame:
    """For each horizon, the lowest-MAE model vs. every other model (common cases per pair).

    Several pairs are tested per horizon, so the Holm-adjusted p-value is also reported. Pairs
    that can't be tested (too few effectively independent cases) are listed with NaN p-values
    and ``testable=False`` rather than silently dropped.
    """
    rows = []
    for horizon, group in forecasts.groupby("horizon"):
        mae = group.assign(abs_error=group["error"].abs()).groupby("model")["abs_error"].mean()
        best = mae.idxmin()
        others = [m for m in mae.index if m != best]
        results = {
            other: compare(forecasts, best, other, int(horizon), step_days=step_days, loss=loss)
            for other in others
        }
        tested = {o: r for o, r in results.items() if r is not None}
        holm = dict(zip(tested, _holm([r.p_value for r in tested.values()]), strict=True))
        for other in others:
            r = tested.get(other)
            rows.append(
                {
                    "horizon": int(horizon),
                    "best": best,
                    "other": other,
                    "testable": r is not None,
                    "n": r.n if r else np.nan,
                    "effective_cases": r.effective_cases if r else np.nan,
                    "mae_diff": r.mean_loss_diff if r else np.nan,
                    "statistic": r.statistic if r else np.nan,
                    "p_value": r.p_value if r else np.nan,
                    "p_holm": holm[other] if r else np.nan,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "horizon", "best", "other", "testable", "n", "effective_cases", "mae_diff",
            "statistic", "p_value", "p_holm",
        ],
    )


def _holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment (NaN p-values pass through as NaN)."""
    p = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(p)
    adjusted = np.full(len(p), np.nan)
    m = int(valid.sum())
    order = np.argsort(np.where(valid, p, np.inf))[:m]
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()
