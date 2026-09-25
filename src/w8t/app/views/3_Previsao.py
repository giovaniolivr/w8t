import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from w8t.app import theme, ui
from w8t.core import metrics
from w8t.data import periods, repository
from w8t.data.db import get_session
from w8t.forecasting.backtest import (
    DEFAULT_HORIZONS,
    DEFAULT_MIN_HISTORY_DAYS,
    walk_forward,
)
from w8t.forecasting.base import InsufficientDataError
from w8t.forecasting.registry import all_models, kalman_holt_ensemble
from w8t.forecasting.significance import MIN_EFFECTIVE_CASES, versus_best

REFERENCE = "Último valor"
FORECAST_DAYS = 30
HISTORY_SHOWN_DAYS = 60
# Every 2 days: ~half the cases of a daily step, but the page loads in ~15 s instead of ~30 s
# with 6 models (cached per series afterwards).
BACKTEST_STEP_DAYS = 2

ui.setup("Previsão")
ui.header(
    "Previsão",
    "Cada modelo é avaliado contra o seu próprio histórico (backtesting walk-forward) antes "
    "de qualquer previsão ser exibida. Previsão é estimativa com incerteza, nunca certeza.",
)

with get_session() as session:
    full_series = metrics.to_series(
        (e.entry_date, e.weight_kg) for e in repository.list_entries(session)
    )
    all_periods = periods.list_periods(session)

if full_series.empty:
    st.info("Nenhum registro de peso ainda.")
    st.stop()

# Scope: the current period (the user-declared regime) by default. On synthetic series with a
# regime switch, forecasting from the current period only cut the 30-day error by 13-14% vs. the
# full history (cutting->bulk 0.48 vs 0.56 kg; cutting->plateau 0.37 vs 0.43); without a switch it
# was ~5% worse (less data) - periods pay off exactly when they mark a real change. A period too
# short for the recommended model falls back to the full history.
current = periods.period_containing(all_periods, full_series.index[-1].date())
period_series = (
    metrics.slice_series(full_series, current.start_date, current.end_date)
    if current is not None else None
)


def _fits(s) -> bool:
    try:
        kalman_holt_ensemble().fit(s)
        return True
    except InsufficientDataError:
        return False


scope_options = {"Histórico completo": full_series}
if current is not None:
    scope_options = {f"Período atual — {current.label}": period_series} | scope_options
period_ok = period_series is not None and _fits(period_series)
scope = st.selectbox(
    "Dados usados",
    list(scope_options),
    index=0 if period_ok else len(scope_options) - 1,
    help="Por padrão, só o período atual: as previsões respeitam a fase que você definiu.",
    key="forecast_scope",
)
series = scope_options[scope]
if current is not None and not period_ok and scope.startswith("Histórico"):
    st.caption(
        f"O período atual tem {len(period_series)} medição(ões) — ainda pouco para o modelo "
        "recomendado; usando o histórico completo."
    )


@st.cache_resource
def _forecast_cache() -> dict:
    """Per-origin forecasts, shared across reruns: after a new entry only new origins are fitted
    (see ``walk_forward``). A few MB at most; cleared if it ever grows past the cap."""
    return {}


FORECAST_CACHE_MAX = 50_000


@st.cache_data(show_spinner="Rodando backtesting...")
def _backtest(s: pd.Series):
    cache = _forecast_cache()
    if len(cache) > FORECAST_CACHE_MAX:
        cache.clear()
    result = walk_forward(s, all_models(), step_days=BACKTEST_STEP_DAYS, cache=cache)
    fc = result.forecasts
    common = fc[fc.groupby(["origin", "horizon"])["model"].transform("nunique")
                == fc["model"].nunique()]
    significance = versus_best(common, step_days=BACKTEST_STEP_DAYS)
    return result.summary(reference=REFERENCE), result.skipped, significance


def _verdict(row) -> str:
    if not row["testable"]:
        return "não testável: poucos casos independentes"
    if row["p_holm"] < 0.01:
        return "diferença demonstrada"
    if row["p_holm"] < 0.05:
        return "evidência fraca"
    return "sem diferença demonstrada"


summary, skipped, significance = _backtest(series)

st.subheader("Avaliação dos modelos (backtesting)")
if summary.empty:
    st.info(
        "Ainda não há histórico suficiente para avaliar os modelos: é preciso pelo menos "
        f"{DEFAULT_MIN_HISTORY_DAYS} dias de registros e medições reais nas datas-alvo "
        f"(horizontes de {', '.join(map(str, DEFAULT_HORIZONS))} dias)."
    )
else:
    table = summary.rename(
        columns={
            "model": "Modelo",
            "horizon": "Horizonte (dias)",
            "n": "Casos",
            "mae": "MAE (kg)",
            "rmse": "RMSE (kg)",
            "bias": "Viés (kg)",
            "coverage": "Cobertura IC95%",
            "mean_width": "Largura média IC (kg)",
            "skill_vs_ref": f"Ganho vs. {REFERENCE}",
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "MAE (kg)": st.column_config.NumberColumn(format="%.2f"),
            "RMSE (kg)": st.column_config.NumberColumn(format="%.2f"),
            "Viés (kg)": st.column_config.NumberColumn(format="%+.2f"),
            "Cobertura IC95%": st.column_config.NumberColumn(format="percent"),
            "Largura média IC (kg)": st.column_config.NumberColumn(format="%.2f"),
            f"Ganho vs. {REFERENCE}": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.caption(
        f"Para cada data passada (a cada {BACKTEST_STEP_DAYS} dias), o modelo é ajustado só com "
        "os dados até aquele dia e comparado com a medição real N dias depois (sem interpolar "
        "dias faltantes). Todos os modelos são comparados nos mesmos casos. **Cobertura** é a "
        "fração de vezes que o valor real caiu dentro do intervalo de 95%: bem abaixo de 95% = "
        "modelo confiante demais; 100% com intervalo largo = cauteloso a ponto de ser pouco "
        "útil. **Viés** positivo = o peso real ficou acima do previsto."
    )

    if not significance.empty:
        st.markdown("**O menor erro é de fato menor?** (teste de Diebold-Mariano)")
        st.dataframe(
            pd.DataFrame(
                {
                    "Horizonte (dias)": significance["horizon"],
                    "Menor MAE": significance["best"],
                    "Comparado com": significance["other"],
                    "Casos independentes": significance["effective_cases"],
                    "p (Holm)": significance["p_holm"],
                    "Conclusão": significance.apply(_verdict, axis=1),
                }
            ),
            hide_index=True,
            column_config={
                "Casos independentes": st.column_config.NumberColumn(format="%.0f"),
                "p (Holm)": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.caption(
            "Previsões com horizonte maior que o intervalo entre origens se sobrepõem no tempo; "
            "o teste leva isso em conta, o que reduz os casos *efetivamente independentes*. "
            f"Abaixo de {MIN_EFFECTIVE_CASES} nada é afirmado. O teste é algo liberal com "
            "dependência forte (≈6-10% de falsos positivos ao nível de 5%), por isso só "
            "p < 0,01 conta como diferença demonstrada."
        )

st.subheader(f"Previsão para os próximos {FORECAST_DAYS} dias")

models = {m.name: m for m in all_models()}
recommended = kalman_holt_ensemble().name
best = summary.groupby("model")["mae"].mean().idxmin() if not summary.empty else None
names = sorted(models, key=lambda n: (n != recommended, n != best))
labels = {n: n for n in names}
labels[recommended] = f"{recommended} (recomendado)"
if best is not None and best != recommended:
    labels[best] = f"{best} (menor erro neste histórico)"


def _can_fit(name: str) -> bool:
    try:
        models[name].fit(series)
        return True
    except InsufficientDataError:
        return False


# Default: the recommended model if the history allows it, else the first one that fits.
default = next((i for i, n in enumerate(names) if _can_fit(n)), 0)
choice = st.selectbox("Modelo", names, index=default, format_func=labels.get,
                      key="forecast_model")
st.caption(
    "**Recomendado** por padrão: no benchmark com 30 séries sintéticas de regimes diferentes "
    "(docs/benchmark.md), a combinação Kalman + Holt foi a única calibrada (~95% de cobertura) "
    "em todos os horizontes e nunca a pior em nenhum regime. O modelo de menor erro no seu "
    "histórico raramente é *demonstradamente* melhor (veja o teste acima)."
)

try:
    forecast = models[choice].fit(series).predict(range(1, FORECAST_DAYS + 1))
except InsufficientDataError as exc:
    st.info(f"Este modelo não pode prever com os dados atuais: {exc}")
    st.stop()

fc = forecast.to_frame()
recent = series[series.index > series.index[-1] - pd.Timedelta(days=HISTORY_SHOWN_DAYS)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=recent.index, y=recent, mode="markers", name="Medição real",
                         marker={"size": 7, "color": theme.MEASUREMENT}))
fig.add_trace(go.Scatter(x=fc.index, y=fc["upper"], mode="lines", line={"width": 0},
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=fc.index, y=fc["lower"], mode="lines", line={"width": 0},
                         fill="tonexty", fillcolor=theme.FORECAST_BAND,
                         name=f"Intervalo {forecast.level:.0%}", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=fc.index, y=fc["mean"], mode="lines", name="Previsão",
                         line={"color": theme.FORECAST, "dash": "dash", "width": 2.5}))
theme.style_figure(fig, height=440, range_slider=False)
st.plotly_chart(fig, width="stretch", config=theme.PLOTLY_CONFIG)

key_rows = fc[fc["horizon_days"].isin(DEFAULT_HORIZONS)]
st.dataframe(
    pd.DataFrame(
        {
            "Data": [ts.strftime("%d/%m/%Y") for ts in key_rows.index],
            "Horizonte (dias)": key_rows["horizon_days"].to_list(),
            "Previsão (kg)": key_rows["mean"].to_list(),
            "Intervalo 95% (kg)": [
                f"{lo:.1f} – {hi:.1f}" for lo, hi in zip(key_rows["lower"], key_rows["upper"],
                                                         strict=True)
            ],
        }
    ),
    hide_index=True,
    column_config={"Previsão (kg)": st.column_config.NumberColumn(format="%.1f")},
)

if not summary.empty:
    measured = summary[summary["model"] == choice].set_index("horizon")["coverage"]
    st.caption(
        "O intervalo é o que as premissas do modelo implicam. No backtesting deste histórico, "
        "a cobertura real foi: "
        + ", ".join(f"{h} d → {c:.0%}" for h, c in measured.items())
        + ". Use isso para calibrar quanto confiar na faixa acima."
    )
