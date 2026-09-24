"""Gaussian Process Regression forecaster (roadmap step 5).

scikit-learn ``GaussianProcessRegressor`` over the trailing ``window_days`` of measurements, with
time as a continuous input (days relative to the origin) - a missing day simply isn't a training
point, nothing is imputed. Kernel::

    C * DotProduct   (linear trend - Bayesian linear regression)
  + C * RBF          (smooth local deviations from that trend)
  + WhiteKernel      (measurement noise)

The predictive std includes the WhiteKernel term, so intervals are for a new *measurement*,
comparable to the other models. Hyperparameters by marginal likelihood (1 optimizer restart).

Kernel/window choice (2026-09-24, demo series, walk-forward step 3 days, vs. damped Holt): pure
RBF or Matern-3/2 kernels revert to the window mean away from the data and were clearly worse at
14-30 days (bias ~ -0.5 kg at h=30: forecasting a rebound); linear + RBF over 60 days was the
best GPR variant. Caveat: selected on the same series it is evaluated on (selection bias) - all
candidates' numbers are recorded in CLAUDE.md.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C

from w8t.core.metrics import slice_series
from w8t.core.trend import days_since
from w8t.forecasting.base import ForecastModel, InsufficientDataError


def _kernel():
    return (
        C(0.1, (1e-3, 1e2)) * DotProduct(1.0, (1e-2, 1e3))
        + C(1.0, (1e-2, 1e2)) * RBF(15.0, (3.0, 365.0))
        + WhiteKernel(0.1, (1e-3, 2.0))
    )


class GPRLinearRBF(ForecastModel):
    min_obs = 14

    def __init__(self, window_days: int = 60) -> None:
        super().__init__()
        self.window_days = window_days
        self.name = f"GPR (linear + RBF, {window_days}d)"

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
                _kernel(), normalize_y=True, n_restarts_optimizer=1, random_state=0
            ).fit(x, y)
        white = self._gp.kernel_.k2
        self.noise_sd = float(np.sqrt(white.noise_level) * self._gp._y_train_std)

    def _predict(self, horizons, level):
        x = horizons.astype(float).reshape(-1, 1)
        mean, sd = self._gp.predict(x, return_std=True)
        half = float(stats.norm.ppf(0.5 + level / 2)) * sd
        return mean, mean - half, mean + half
