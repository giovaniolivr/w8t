"""Equal-weight combination of forecasters.

Motivation (docs/benchmark.md, 2026-09-24): the Kalman smooth trend wins when a trend keeps going,
the damped Holt wins when the regime changes (the damping is a bet that trends end), and which
regime a user is in is unknown at forecast time. Averaging two models that fail in opposite
situations is the classic way to be robust to that.

- Mean: average of the members' point forecasts.
- Interval, two options:
  - ``"mixture"`` - the 50/50 mixture of the members' Gaussian predictive distributions; its
    quantiles are solved numerically. When members *disagree*, the mixture widens: model
    disagreement is treated as uncertainty.
  - ``"quantile_avg"`` - average of the members' interval bounds (narrower; ignores
    disagreement).
  Members' intervals are symmetric Gaussian ones, so each member's sd is recovered as
  ``(upper - lower) / (2 z)``.

Gate: the ensemble fits only if *every* member can - no silently degrading to a single model.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

from w8t.forecasting.base import ForecastModel


def _mixture_quantile(q: float, means: np.ndarray, sds: np.ndarray) -> float:
    def cdf_gap(x: float) -> float:
        return float(np.mean(stats.norm.cdf(x, means, sds))) - q

    lo = float(np.min(means - 10 * sds))
    hi = float(np.max(means + 10 * sds))
    return optimize.brentq(cdf_gap, lo, hi, xtol=1e-6)


class EqualWeightEnsemble(ForecastModel):
    def __init__(
        self, members: Sequence[ForecastModel], *, interval: str = "mixture", name: str
    ) -> None:
        super().__init__()
        if interval not in ("mixture", "quantile_avg"):
            raise ValueError(f"interval desconhecido: {interval}")
        self.members = list(members)
        self.interval = interval
        self.name = name
        self.min_obs = max(m.min_obs for m in self.members)

    @property
    def member_names(self) -> list[str]:
        return [m.name for m in self.members]

    def _fit(self, series: pd.Series) -> None:
        # Fresh copies so fitting the ensemble never mutates the prototypes it was built from.
        self._fitted = [copy.deepcopy(m).fit(series) for m in self.members]

    def fit_from_fitted(self, fitted_members: Sequence[ForecastModel]) -> EqualWeightEnsemble:
        """Combine members already fitted on the *same* series (backtesting reuses them instead
        of refitting); equivalent to ``fit`` on that series."""
        origins = {m._origin for m in fitted_members}
        if len(origins) != 1 or None in origins:
            raise ValueError("Membros precisam estar ajustados na mesma série.")
        if [m.name for m in fitted_members] != self.member_names:
            raise ValueError("Membros não correspondem aos do ensemble.")
        self._fitted = list(fitted_members)
        self._origin = origins.pop()
        return self

    def _predict(self, horizons, level):
        forecasts = [m.predict(horizons, level=level) for m in self._fitted]
        means = np.vstack([f.mean for f in forecasts])  # members x horizons
        mean = means.mean(axis=0)
        if self.interval == "quantile_avg":
            lower = np.vstack([f.lower for f in forecasts]).mean(axis=0)
            upper = np.vstack([f.upper for f in forecasts]).mean(axis=0)
            return mean, lower, upper

        z = float(stats.norm.ppf(0.5 + level / 2))
        sds = np.vstack([(f.upper - f.lower) / (2 * z) for f in forecasts])
        tail = (1 - level) / 2
        lower = np.array(
            [_mixture_quantile(tail, means[:, j], sds[:, j]) for j in range(means.shape[1])]
        )
        upper = np.array(
            [_mixture_quantile(1 - tail, means[:, j], sds[:, j]) for j in range(means.shape[1])]
        )
        return mean, lower, upper
