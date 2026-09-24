"""Walk-forward backtesting (spec item 15).

For each forecast origin (a real measurement date), every model is refit on a *fresh copy*
using only measurements up to and including the origin, then asked for the configured
horizons. A forecast is scored only if a real measurement exists exactly on the target date -
gaps are never interpolated to manufacture a "truth". Nothing is shuffled; no information after
the origin reaches the model.

Summary metrics per (model, horizon): MAE, RMSE, bias (mean of truth - forecast), empirical
interval coverage (should be close to the nominal level - far below means over-confident, far
above means uselessly wide), mean interval width, and skill vs. a reference model
(1 - MAE / MAE_ref; > 0 means better than the reference). By default metrics are computed only
on the cases every model forecast, so models are compared on identical ground.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from w8t.forecasting.base import ForecastModel, InsufficientDataError

DEFAULT_HORIZONS = (1, 7, 14, 30)
DEFAULT_STEP_DAYS = 7
DEFAULT_MIN_HISTORY_DAYS = 28


@dataclass(frozen=True)
class BacktestResult:
    forecasts: pd.DataFrame  # one row per scored (model, origin, horizon)
    skipped: pd.DataFrame  # one row per (model, origin) the model refused to fit
    level: float

    def summary(self, reference: str | None = None, *, common_only: bool = True) -> pd.DataFrame:
        return summarize(self.forecasts, reference=reference, common_only=common_only)


def forecast_origins(
    series: pd.Series, step_days: int, min_history_days: int
) -> list[pd.Timestamp]:
    """Measurement dates at least ``min_history_days`` after the first one and at least
    ``step_days`` apart."""
    if series.empty:
        return []
    first = series.index[0]
    origins: list[pd.Timestamp] = []
    for ts in series.index:
        if (ts - first).days < min_history_days:
            continue
        if origins and (ts - origins[-1]).days < step_days:
            continue
        origins.append(ts)
    return origins


def walk_forward(
    series: pd.Series,
    models: Sequence[ForecastModel],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    *,
    step_days: int = DEFAULT_STEP_DAYS,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
    level: float = 0.95,
) -> BacktestResult:
    rows: list[dict] = []
    skipped: list[dict] = []

    for origin in forecast_origins(series, step_days, min_history_days):
        train = series[series.index <= origin]
        targets = {h: origin + pd.Timedelta(days=h) for h in horizons}
        truths = {h: series.get(t) for h, t in targets.items()}
        if all(v is None for v in truths.values()):
            continue  # nothing to score from this origin

        fitted: dict[str, ForecastModel] = {}  # this origin's fits, reusable by ensembles
        for prototype in models:
            model = copy.deepcopy(prototype)
            try:
                members = getattr(prototype, "member_names", None)
                if members and all(name in fitted for name in members):
                    model.fit_from_fitted([fitted[name] for name in members])
                else:
                    model.fit(train)
                fc = model.predict(list(horizons), level=level)
            except InsufficientDataError as exc:
                skipped.append({"model": prototype.name, "origin": origin, "reason": str(exc)})
                continue
            fitted[prototype.name] = model
            for i, h in enumerate(fc.horizons):
                y = truths[int(h)]
                if y is None:
                    continue
                rows.append(
                    {
                        "model": fc.model,
                        "origin": origin,
                        "horizon": int(h),
                        "target_date": targets[int(h)],
                        "y_true": float(y),
                        "mean": fc.mean[i],
                        "lower": fc.lower[i],
                        "upper": fc.upper[i],
                    }
                )

    forecasts = pd.DataFrame(
        rows,
        columns=["model", "origin", "horizon", "target_date", "y_true", "mean", "lower", "upper"],
    )
    forecasts["error"] = forecasts["y_true"] - forecasts["mean"]
    forecasts["covered"] = (forecasts["lower"] <= forecasts["y_true"]) & (
        forecasts["y_true"] <= forecasts["upper"]
    )
    forecasts["width"] = forecasts["upper"] - forecasts["lower"]
    return BacktestResult(
        forecasts=forecasts,
        skipped=pd.DataFrame(skipped, columns=["model", "origin", "reason"]),
        level=level,
    )


def summarize(
    forecasts: pd.DataFrame, reference: str | None = None, *, common_only: bool = True
) -> pd.DataFrame:
    """Per (model, horizon) metrics.

    ``common_only``: keep only (origin, horizon) cases that *every* model forecast. Models gate
    themselves differently, and comparing MAE over different case sets would be unfair (a model
    that skips hard early origins would look better for free).
    """
    if common_only and not forecasts.empty:
        n_models = forecasts["model"].nunique()
        per_case = forecasts.groupby(["origin", "horizon"])["model"].transform("nunique")
        forecasts = forecasts[per_case == n_models]
    if forecasts.empty:
        return pd.DataFrame(
            columns=["model", "horizon", "n", "mae", "rmse", "bias", "coverage", "mean_width"]
        )
    grouped = forecasts.groupby(["model", "horizon"], sort=False)
    out = grouped.agg(
        n=("error", "size"),
        mae=("error", lambda e: float(np.mean(np.abs(e)))),
        rmse=("error", lambda e: float(np.sqrt(np.mean(e**2)))),
        bias=("error", "mean"),
        coverage=("covered", "mean"),
        mean_width=("width", "mean"),
    ).reset_index()

    if reference is not None:
        ref = out[out["model"] == reference].set_index("horizon")["mae"]
        out["skill_vs_ref"] = 1 - out["mae"] / out["horizon"].map(ref)
    return out.sort_values(["horizon", "model"], kind="stable").reset_index(drop=True)
