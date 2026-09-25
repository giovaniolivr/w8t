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

Optional ``cache`` (any mutable mapping): a forecast made at an origin depends only on the model
and the training data up to that origin, so it is stored under (model name, level, horizons,
hash of the training data). When the user adds today's entry, every past origin is a cache hit
and only the new origins are fitted - the backtest no longer restarts from scratch. A retroactive
entry or edit changes the hash of every later training set, so those are refitted. Model names
must identify the configuration (they do in the registry).
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import MutableMapping, Sequence
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
    cache: MutableMapping | None = None,
) -> BacktestResult:
    rows: list[dict] = []
    skipped: list[dict] = []

    for origin in forecast_origins(series, step_days, min_history_days):
        train = series[series.index <= origin]
        targets = {h: origin + pd.Timedelta(days=h) for h in horizons}
        truths = {h: series.get(t) for h, t in targets.items()}
        if all(v is None for v in truths.values()):
            continue  # nothing to score from this origin

        train_key = _data_key(train) if cache is not None else None
        fitted: dict[str, ForecastModel] = {}  # this origin's fits, reusable by ensembles
        for prototype in models:
            key = (prototype.name, level, tuple(horizons), train_key)
            fc = cache.get(key) if cache is not None else None
            if fc is None:
                fc = _fit_predict(prototype, train, fitted, horizons, level)
                if cache is not None:
                    cache[key] = fc
            if isinstance(fc, str):  # the model refused this training set
                skipped.append({"model": prototype.name, "origin": origin, "reason": fc})
                continue
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


def _fit_predict(prototype, train, fitted, horizons, level):
    """Forecast from a fresh copy of ``prototype``, or the reason it refused to fit."""
    model = copy.deepcopy(prototype)
    try:
        members = getattr(prototype, "member_names", None)
        if members and all(name in fitted for name in members):
            model.fit_from_fitted([fitted[name] for name in members])
        else:
            model.fit(train)
        fc = model.predict(list(horizons), level=level)
    except InsufficientDataError as exc:
        return str(exc)
    fitted[prototype.name] = model
    return fc


def _data_key(series: pd.Series) -> str:
    h = hashlib.sha1(series.index.asi8.tobytes())
    h.update(np.ascontiguousarray(series.to_numpy(dtype=float)).tobytes())
    return h.hexdigest()


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
