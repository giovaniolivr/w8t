"""Kalman filter over a structural trend model (roadmap step 4).

``statsmodels.tsa.UnobservedComponents`` with a *smooth trend*: the observed weight is an
underlying level plus measurement noise, and the level moves with a slope that itself follows a
random walk::

    y_t      = mu_t + eps_t,           eps_t  ~ N(0, sigma2_irregular)   (scale noise, water...)
    mu_{t+1} = mu_t + beta_t
    beta_t+1 = beta_t + zeta_t,        zeta_t ~ N(0, sigma2_trend)       (how fast the pace changes)

Unlike ETS, the Kalman filter handles missing days natively: on a daily grid, a NaN day is a
prediction step with no update - verified to fit/converge with gaps (2026-09-24). Forecast
variance from ``get_forecast`` includes the measurement noise, i.e. it is a prediction interval
for a future *measurement*, comparable to the other models.

Only the forward *filter* is used for forecasting and backtesting (state at the origin uses data
up to the origin). The RTS *smoother* - which uses the whole series - is for describing the past
only and must never feed an evaluation.

Why smooth trend and not the full local linear trend: on the demo series both reach the same
likelihood (the extra level-noise variance is estimated ~0), the smooth trend has a better AIC
and fits ~6x faster, which matters because backtesting refits at every origin.

Interval calibration at long horizons (investigated 2026-09-25). On the 30-series benchmark the
h=30 coverage is 86%, but the shortfall is concentrated in regime-change scenarios (cutting->bulk
67%, cutting->plateau 72%; ~95% elsewhere): fitted on the whole series the slope volatility
``sigma2.trend`` comes out ~0, so the current pace is treated as known. A floor on that
volatility was tested on 18 held-out series (floors 0.002-0.016 kg/day): reaching 95% overall
needed 0.008, which made intervals 2x wider and 100% covering on steady scenarios, still left
cutting->bulk at 81%, and raised h=30 MAE 0.59 -> 0.68 - rejected. An unannounced regime change
is not something any of these models can foresee. What does work is the user declaring it:
fitting only on the current period (what the forecast page does), from 25 days after the switch,
h=30 on cutting->bulk / cutting->plateau (10 held-out series): coverage 92%, MAE 0.60 -> 0.49,
interval 6.2 -> 2.2 kg vs. the full history.

Weekly pattern (added 2026-09-24): many people weigh more after weekends. An optional fixed
7-day seasonal component (``seasonal=7``, deterministic) captures it - but only when the data
supports it, otherwise it would fit noise. With ``weekly=None`` (default) both models are fitted
and the weekly one is kept only if its one-step-ahead predictive log-likelihood, scored on the
same observations after a 21-day burn-in, is higher (``WEEKLY_MIN_GAIN``). Only tried with
>= ``WEEKLY_MIN_SPAN_DAYS`` of data. Measured (120-day series, noise 0.35 kg, 15% missing): 0/30
false selections without a pattern; weekend bump of 0.5 kg selected 97%, 0.35 kg 80%, 0.25 kg
37%, 0.15 kg 10%. On the 6 labeled scenarios x 8 seeds: 8/8 on the weekly one, 0/40 elsewhere.
The level state (index 0) stays the deseasonalized weight; the slope stays index 1.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

from w8t.core.trend import days_since
from w8t.forecasting.base import ForecastModel, InsufficientDataError


class NotConvergedError(InsufficientDataError):
    pass


WEEKLY_MIN_SPAN_DAYS = 56
WEEKLY_MIN_GAIN = 0.0  # nats; the predictive score already penalizes the extra states
SCORE_BURN_IN_DAYS = 21


def fit_smooth_trend(series: pd.Series, weekly: bool | None = None):
    """Fit the smooth-trend model on a daily grid (missing days = NaN, never filled).

    ``weekly``: True/False forces the weekly component; None chooses it from the data (see the
    module docstring). Returns the statsmodels results object; ``has_weekly(result)`` tells which
    one was kept. Raises :class:`NotConvergedError` if the fit doesn't converge.
    """
    if weekly is not None:
        return _fit(series, weekly)
    plain = _fit(series, False)
    if (series.index[-1] - series.index[0]).days < WEEKLY_MIN_SPAN_DAYS:
        return plain
    try:
        seasonal = _fit(series, True)
    except NotConvergedError:
        return plain
    gain = _predictive_score(seasonal) - _predictive_score(plain)
    return seasonal if gain > WEEKLY_MIN_GAIN else plain


def has_weekly(result) -> bool:
    return bool(getattr(result.model, "seasonal", False))


def smoothed_signal(result) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed expected measurement (level + weekly effect, without scale noise) and its
    variance, for every day of the grid. Without the weekly component this is just the level."""
    design = result.filter_results.design[0, :, 0]  # time-invariant observation vector
    states = result.smoothed_state
    cov = result.smoothed_state_cov
    mean = design @ states
    var = np.einsum("i,ijt,j->t", design, cov, design)
    return mean, np.clip(var, 0.0, None)


def weekly_effect_range(result) -> float | None:
    """Peak-to-trough size of the estimated weekly pattern (kg), None without it."""
    if not has_weekly(result):
        return None
    mean, _ = smoothed_signal(result)
    effect = mean - result.smoothed_state[0]
    last_week = effect[-7:]
    return float(last_week.max() - last_week.min())


def _predictive_score(result) -> float:
    """Gaussian log-likelihood of the one-step-ahead predictions of observed days after the
    burn-in - comparable between models because it's scored on the same observations."""
    fr = result.filter_results
    v, f = fr.forecasts_error[0], fr.forecasts_error_cov[0, 0]
    scored = ~np.isnan(v)
    scored[:SCORE_BURN_IN_DAYS] = False
    return float(np.sum(-0.5 * (np.log(2 * np.pi * f[scored]) + v[scored] ** 2 / f[scored])))


def _fit(series: pd.Series, weekly: bool):
    daily = series.asfreq("D")
    extra = {"seasonal": 7, "stochastic_seasonal": False} if weekly else {}
    model = UnobservedComponents(daily, level="strend", **extra)
    with warnings.catch_warnings():
        # statsmodels warns liberally; convergence is checked explicitly below
        warnings.simplefilter("ignore")
        result = model.fit(disp=False)
        if not result.mle_retvals.get("converged", False):
            # L-BFGS flags "not converged" when the optimum sits on the boundary
            # (sigma2.trend -> 0, a deterministic trend). Powell reaches the same likelihood
            # and parameters and does report convergence there (checked on the demo), so
            # retry rather than discard a valid estimate.
            result = model.fit(disp=False, method="powell", maxiter=2000)
    if not result.mle_retvals.get("converged", False):
        raise NotConvergedError("Kalman: estimação não convergiu com estes dados.")
    return result


class KalmanSmoothTrend(ForecastModel):
    name = "Kalman (tendência suave)"
    min_obs = 14
    min_span_days = 21

    def _fit(self, series: pd.Series) -> None:
        if days_since(series.index, series.index[0])[-1] < self.min_span_days:
            raise InsufficientDataError(
                f"{self.name}: precisa de medições cobrindo ≥ {self.min_span_days} dias."
            )
        result = fit_smooth_trend(series)
        self._result = result
        self.weekly = has_weekly(result)
        params = dict(zip(result.model.param_names, result.params, strict=True))
        self.noise_sd = float(np.sqrt(params["sigma2.irregular"]))
        self.slope_change_sd = float(np.sqrt(params["sigma2.trend"]))

    def _predict(self, horizons, level):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self._result.get_forecast(int(horizons.max()))
            ci = fc.conf_int(alpha=1 - level).to_numpy()
        idx = horizons - 1
        return fc.predicted_mean.to_numpy()[idx], ci[idx, 0], ci[idx, 1]
