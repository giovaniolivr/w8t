"""Gaussian Process Regression forecaster (roadmap step 5).

scikit-learn ``GaussianProcessRegressor`` over the trailing ``window_days`` of measurements, with
time as a continuous input (days relative to the origin) - a missing day simply isn't a training
point, nothing is imputed. Kernel::

    C * DotProduct   (linear trend - Bayesian linear regression)
  + C * RBF          (smooth local deviations from that trend)
  + WhiteKernel      (measurement noise)

The predictive std includes the WhiteKernel term, so intervals are for a new *measurement*,
comparable to the other models. Hyperparameters by marginal likelihood (1 optimizer restart).

Kernel/window choice. First pass (2026-09-24): picked linear + RBF 60d by looking at the demo
series only - on the multi-series benchmark it then ranked last of the top four models (selection
bias). Second pass, done properly: 8 variants (linear + {RBF, Matern 3/2, RQ, RBF + fixed 7-day
periodic} x {60, 90} days) compared on 18 *held-out* series (seeds 100-102, walk-forward every 5
days), never on the benchmark seeds. Winner: linear + Matern 3/2, 90 days (mean MAE 0.492 vs 0.526
for the old variant; h=30 coverage 85% vs 80.5%). The weekly periodic kernel didn't help overall
(only 1 of 6 scenarios has a weekly pattern; the Kalman handles it with automatic selection).
Confirmation on the 30 benchmark series (seeds 0-4): lower MAE at every horizon (h=30 0.738 vs
0.762, coverage 85% vs 82%) but NOT statistically significant (Wilcoxon p = 0.27-0.38). Same
regime trade-off as elsewhere: the longer window wins on continuing trends (bulk h=30 0.46 vs
0.59) and loses right after reversals (cutting->plateau 0.92 vs 0.73). Still behind the Kalman and
the Kalman + Holt combination.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    DotProduct,
    ExpSineSquared,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.gaussian_process.kernels import ConstantKernel as C

from w8t.core.metrics import slice_series
from w8t.core.trend import days_since
from w8t.forecasting.base import ForecastModel, InsufficientDataError


def _linear():
    return C(0.1, (1e-3, 1e2)) * DotProduct(1.0, (1e-2, 1e3))


def _noise():
    return WhiteKernel(0.1, (1e-3, 2.0))


# Candidate kernels (all: linear trend + smooth local deviations + measurement noise, noise last).
KERNELS = {
    "linear + RBF": lambda: _linear() + C(1.0, (1e-2, 1e2)) * RBF(15.0, (3.0, 365.0)) + _noise(),
    "linear + Matérn 3/2": lambda: (
        _linear() + C(1.0, (1e-2, 1e2)) * Matern(15.0, (3.0, 365.0), nu=1.5) + _noise()
    ),
    "linear + RQ": lambda: (
        _linear()
        + C(1.0, (1e-2, 1e2)) * RationalQuadratic(15.0, 1.0, (3.0, 365.0), (1e-2, 1e2))
        + _noise()
    ),
    # Weekly pattern: a periodic component with the period fixed at 7 days.
    "linear + RBF + semanal": lambda: (
        _linear()
        + C(1.0, (1e-2, 1e2)) * RBF(15.0, (3.0, 365.0))
        + C(0.1, (1e-3, 1e1)) * ExpSineSquared(1.0, 7.0, (0.1, 10.0), "fixed")
        + _noise()
    ),
}
DEFAULT_KERNEL = "linear + RBF"


def _kernel(name: str = DEFAULT_KERNEL):
    return KERNELS[name]()


class GPR(ForecastModel):
    min_obs = 14

    def __init__(self, kernel: str = DEFAULT_KERNEL, window_days: int = 60) -> None:
        super().__init__()
        if kernel not in KERNELS:
            raise ValueError(f"Kernel desconhecido: {kernel}")
        self.kernel = kernel
        self.window_days = window_days
        self.name = f"GPR ({kernel}, {window_days}d)"

    def _fit(self, series: pd.Series) -> None:
        end = series.index[-1]
        window = slice_series(
            series, (end - pd.Timedelta(days=self.window_days - 1)).date(), end.date()
        )
        if len(window) < self.min_obs:
            raise InsufficientDataError(
                f"{self.name}: precisa de ≥ {self.min_obs} medições nos últimos "
                f"{self.window_days} dias, há {len(window)}."
            )
        x = days_since(window.index, end).reshape(-1, 1)  # origin at 0, past negative
        y = window.to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # hyperparameters at a bound are acceptable here
            self._gp = GaussianProcessRegressor(
                _kernel(self.kernel), normalize_y=True, n_restarts_optimizer=1, random_state=0
            ).fit(x, y)
        white = self._gp.kernel_.k2  # the noise term is always the last summand
        self.noise_sd = float(np.sqrt(white.noise_level) * self._gp._y_train_std)

    def _predict(self, horizons, level):
        x = horizons.astype(float).reshape(-1, 1)
        mean, sd = self._gp.predict(x, return_std=True)
        half = float(stats.norm.ppf(0.5 + level / 2)) * sd
        return mean, mean - half, mean + half


class GPRLinearRBF(GPR):
    """The original variant (kept for existing callers)."""

    def __init__(self, window_days: int = 60) -> None:
        super().__init__(DEFAULT_KERNEL, window_days)
