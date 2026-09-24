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
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

from w8t.core.trend import days_since
from w8t.forecasting.base import ForecastModel, InsufficientDataError


class KalmanSmoothTrend(ForecastModel):
    name = "Kalman (tendência suave)"
    min_obs = 14
    min_span_days = 21

    def _fit(self, series: pd.Series) -> None:
        if days_since(series.index, series.index[0])[-1] < self.min_span_days:
            raise InsufficientDataError(
                f"{self.name}: precisa de medições cobrindo ≥ {self.min_span_days} dias."
            )
        daily = series.asfreq("D")  # missing days become NaN - handled by the filter, not filled
        model = UnobservedComponents(daily, level="strend")
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
            raise InsufficientDataError(f"{self.name}: estimação não convergiu com estes dados.")
        self._result = result
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
