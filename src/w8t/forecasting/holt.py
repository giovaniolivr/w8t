"""Holt's linear method with damped trend - ETS(A,Ad,N) - with native missing-day handling.

Why not statsmodels: both ``ExponentialSmoothing`` and ``ETSModel`` require a regular series.
Fed a daily series with NaN gaps they fail *silently* (no convergence / NaN likelihood, all
fitted values NaN - verified 2026-09-24), and resampling + interpolating first would fabricate
measurements. So this is a small, explicit state-space implementation:

- Daily grid from first to last measurement. Each day: one-step prediction
  ``l + phi*b``. Observed day -> error correction; missing day -> state just propagates
  (``l <- l + phi*b``, ``b <- phi*b``). Nothing is imputed.
- Error correction form (Hyndman & Athanasopoulos): ``l_t = l_{t-1} + phi*b_{t-1} + alpha*e_t``,
  ``b_t = phi*b_{t-1} + beta*e_t``, with ``beta = alpha * beta_star`` so ``0 < beta < alpha``.
- An error observed ``k`` days after the previous measurement is a k-step forecast error, with
  variance ``sigma^2 * v(k)``, ``v(k) = 1 + sum_{j=1}^{k-1} c_j^2``,
  ``c_j = alpha + beta*(phi + ... + phi^j)``. Parameters maximize the Gaussian likelihood with
  that per-error variance (sigma^2 concentrated out) - gaps are accounted for, not ignored.
- Forecast at horizon h: mean ``l + (phi + ... + phi^h)*b``, variance ``sigma^2 * v(h)``.

Damping (phi < 1) makes the trend fade instead of extrapolating forever - the specific failure
of the linear-trend baseline at plateaus.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from w8t.core.trend import days_since
from w8t.forecasting.base import ForecastModel, InsufficientDataError

ALPHA_BOUNDS = (0.01, 0.99)
BETA_STAR_BOUNDS = (0.001, 0.99)
PHI_BOUNDS = (0.80, 0.98)  # usual range: below 0.8 damping is too strong to be a trend
INIT_TREND_DAYS = 14
STARTS = ((0.3, 0.1, 0.9), (0.1, 0.05, 0.95))


def _damped_sums(phi: float, n: int) -> np.ndarray:
    """phi_j = phi + phi^2 + ... + phi^j for j = 0..n (phi_0 = 0)."""
    return np.concatenate([[0.0], np.cumsum(phi ** np.arange(1, n + 1))])


def _variance_factors(alpha: float, beta: float, phi: float, max_k: int) -> np.ndarray:
    """v(k) for k = 0..max_k (index k); v(1) = 1."""
    c = alpha + beta * _damped_sums(phi, max_k)[1:max_k]  # c_1 .. c_{max_k - 1}
    v = np.ones(max_k + 1)
    v[2:] = 1 + np.cumsum(c**2)
    return v


def holt_filter(
    y: np.ndarray, level: float, trend: float, alpha: float, beta: float, phi: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Run the damped-Holt recursion over a daily grid (NaN = no measurement that day).

    ``level``/``trend`` are the state at day 0 (``y[0]`` is not scored). Returns the forecast
    error at each observed day after day 0, the gap in days since the previous observed day,
    and the final state.
    """
    errors, gaps = [], []
    since_obs = 0
    for t in range(1, len(y)):
        since_obs += 1
        pred = level + phi * trend
        if np.isnan(y[t]):
            level, trend = pred, phi * trend
            continue
        e = y[t] - pred
        errors.append(e)
        gaps.append(since_obs)
        since_obs = 0
        level, trend = pred + alpha * e, phi * trend + beta * e
    return np.array(errors), np.array(gaps, dtype=int), level, trend


class HoltDamped(ForecastModel):
    name = "Holt amortecido"
    min_obs = 14
    min_span_days = 21

    def _fit(self, series: pd.Series) -> None:
        days = days_since(series.index, series.index[0]).astype(int)
        if days[-1] < self.min_span_days:
            raise InsufficientDataError(
                f"{self.name}: precisa de medições cobrindo ≥ {self.min_span_days} dias."
            )
        y = np.full(days[-1] + 1, np.nan)
        y[days] = series.to_numpy()

        init_mask = days <= INIT_TREND_DAYS
        l0 = float(series.iloc[0])
        b0 = float(np.polyfit(days[init_mask], series.to_numpy()[init_mask], 1)[0])

        def run(params):
            alpha, beta_star, phi = params
            beta = alpha * beta_star
            errors, gaps, level, trend = holt_filter(y, l0, b0, alpha, beta, phi)
            return errors, gaps, level, trend, beta

        def neg_loglik(params):
            errors, gaps, _, _, beta = run(params)
            v = _variance_factors(params[0], beta, params[2], int(gaps.max()))[gaps]
            sigma2 = np.mean(errors**2 / v)
            return len(errors) * np.log(sigma2) + np.sum(np.log(v))

        bounds = [ALPHA_BOUNDS, BETA_STAR_BOUNDS, PHI_BOUNDS]
        best = min(
            (optimize.minimize(neg_loglik, x0, bounds=bounds, method="L-BFGS-B") for x0 in STARTS),
            key=lambda r: r.fun,
        )
        alpha, _, phi = best.x
        errors, gaps, level, trend, beta = run(best.x)
        v = _variance_factors(alpha, beta, phi, int(gaps.max()))[gaps]

        self.alpha, self.beta, self.phi = float(alpha), float(beta), float(phi)
        self._level, self._trend = float(level), float(trend)
        self._sigma = float(np.sqrt(np.mean(errors**2 / v)))

    def _predict(self, horizons, level):
        max_h = int(horizons.max())
        mean = self._level + _damped_sums(self.phi, max_h)[horizons] * self._trend
        v = _variance_factors(self.alpha, self.beta, self.phi, max_h)[horizons]
        half = float(stats.norm.ppf(0.5 + level / 2)) * self._sigma * np.sqrt(v)
        return mean, mean - half, mean + half
