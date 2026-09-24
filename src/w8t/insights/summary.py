"""Structured summary: the *only* input the LLM ever gets.

Everything here is already computed by the other layers (metrics, pattern detectors, forecasting)
and rounded. The raw series is never included. Time is expressed as "days ago" (relative to
``today``) instead of calendar dates: it reads naturally, and it keeps the fixed demo narrative
valid, since the demo data is anchored to the current date.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from w8t.core import metrics
from w8t.data.models import GoalDirection, Period
from w8t.forecasting.base import InsufficientDataError
from w8t.forecasting.registry import kalman_holt_ensemble
from w8t.patterns import gaps, pipeline

GOALS = {
    GoalDirection.LOSS: "perda de peso",
    GoalDirection.GAIN: "ganho de peso",
    GoalDirection.MAINTENANCE: "manutenção",
}
FORECAST_HORIZONS = (7, 30)
INSUFFICIENT = "dados insuficientes"


def _ago(d: date | pd.Timestamp, today: date) -> int:
    d = d.date() if isinstance(d, pd.Timestamp) else d
    return (today - d).days


def _kg(x: float) -> float:
    return round(float(x), 1)


def build_summary(
    series: pd.Series,
    periods: list[Period],
    scope: Period | None,
    today: date,
) -> dict:
    """``series`` is the full history; ``scope`` restricts the analysis to one period."""
    if scope is not None:
        series = metrics.slice_series(series, scope.start_date, scope.end_date or today)
    if series.empty:
        return {"escopo": _scope(scope, today), "registros": INSUFFICIENT}

    summary = metrics.summarize(series)
    detected = pipeline.detect(series)
    found_gaps = gaps.find_gaps(series)

    out: dict = {
        "escopo": _scope(scope, today),
        "registros": {
            "medicoes": summary.n_entries,
            "dias_de_calendario": summary.span_days + 1,
            "primeira_medicao_ha_dias": _ago(summary.first_date, today),
            "ultima_medicao_ha_dias": _ago(summary.last_date, today),
            "lacunas_entre_registros": len(found_gaps),
            "maior_lacuna_dias": max((g.days for g in found_gaps), default=0),
        },
        "peso": {
            "inicial_kg": _kg(summary.initial_kg),
            "atual_kg": _kg(summary.current_kg),
            "minimo_kg": _kg(summary.min_kg),
            "minimo_ha_dias": _ago(summary.min_date, today),
            "maximo_kg": _kg(summary.max_kg),
            "maximo_ha_dias": _ago(summary.max_date, today),
            "variacao_kg": _kg(summary.change_kg),
            "variacao_percentual": round(summary.change_pct, 1),
            "media_movel_7d_kg": (
                _kg(summary.moving_avg_7d) if summary.moving_avg_7d is not None else INSUFFICIENT
            ),
        },
        "tendencia_atual": _trend(detected),
        "platos": [
            {
                "inicio_ha_dias": _ago(p.start, today),
                "fim_ha_dias": _ago(p.end, today),
                "duracao_dias": p.days,
                "peso_medio_kg": _kg(p.mean_kg),
                "periodos_do_usuario": [
                    f"{q.label} ({GOALS[q.goal_direction]})"
                    for q in periods
                    if q.start_date <= p.end and p.start <= (q.end_date or date.max)
                ],
            }
            for p in detected.plateaus
        ],
        "medicoes_atipicas": [
            {
                "ha_dias": _ago(ts, today),
                "peso_kg": _kg(row.weight_kg),
                "esperado_pela_tendencia_kg": _kg(row.expected_kg),
            }
            for ts, row in detected.anomalies[detected.anomalies["is_anomaly"]].iterrows()
        ],
    }
    out |= _forecast(series, scope, today)
    if scope is not None and scope.target_weight_kg is not None:
        out["meta"] = {
            "meta_kg": _kg(scope.target_weight_kg),
            "diferenca_da_ultima_medicao_ate_a_meta_kg": _kg(
                scope.target_weight_kg - summary.current_kg
            ),
        }
    return out


def _scope(scope: Period | None, today: date) -> dict:
    if scope is None:
        return {"tipo": "histórico completo"}
    return {
        "tipo": "período definido pelo usuário",
        "nome": scope.label,
        "objetivo": GOALS[scope.goal_direction],
        "inicio_ha_dias": _ago(scope.start_date, today),
        "situacao": "em andamento" if scope.end_date is None else (
            f"encerrado há {_ago(scope.end_date, today)} dias"
        ),
    }


def _trend(detected: pipeline.Patterns) -> dict | str:
    t = detected.trend
    if t is None:
        return INSUFFICIENT
    return {
        "direcao": t.direction.value,
        "ritmo_kg_por_semana": round(t.slope_kg_per_week, 2),
        "intervalo_95_kg_por_semana": [
            round(t.ci_low_kg_per_week, 2), round(t.ci_high_kg_per_week, 2)
        ],
        "metodo": (
            "filtro de Kalman (últimos 60 dias)" if detected.trend_method == "kalman"
            else "regressão linear (últimos 21 dias)"
        ),
        "faixa_considerada_estavel_kg_por_semana": 0.25,
    }


def _forecast(series: pd.Series, scope: Period | None, today: date) -> dict:
    if scope is not None and scope.end_date is not None:
        return {"previsao": "não se aplica: período encerrado"}
    model = kalman_holt_ensemble()
    try:
        fc = model.fit(series).predict(list(FORECAST_HORIZONS))
    except InsufficientDataError as exc:
        return {"previsao": f"{INSUFFICIENT} ({exc})"}
    kalman = model._fitted[0]
    return {
        "previsao": {
            "modelo": model.name,
            "observacao": (
                "estimativa com intervalo de 95%; em avaliação com 30 séries sintéticas o "
                "intervalo deste modelo conteve o valor real cerca de 95% das vezes"
            ),
            "horizontes": [
                {
                    "dias_apos_a_ultima_medicao": int(h),
                    "peso_previsto_kg": _kg(m),
                    "intervalo_95_kg": [_kg(lo), _kg(hi)],
                }
                for h, m, lo, hi in zip(fc.horizons, fc.mean, fc.lower, fc.upper, strict=True)
            ],
        },
        "ruido_tipico_da_balanca_kg": round(kalman.noise_sd, 2),
    }
