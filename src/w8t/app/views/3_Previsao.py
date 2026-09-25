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
controls = st.container(border=True, key="w8t-card-forecast-controls")
scope_col, scope_box = controls.columns(2)
scope = scope_col.selectbox(
    "Dados usados",
    list(scope_options),
    index=0 if period_ok else len(scope_options) - 1,
    help="Por padrão, só o período atual: as previsões respeitam a fase que você definiu.",
    key="forecast_scope",
)
series = scope_options[scope]
if current is not None and not period_ok and scope.startswith("Histórico"):
    controls.caption(
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

# --- forecast first: it's what the page is for; the evaluation that justifies it comes after ---
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
choice = scope_box.selectbox("Modelo", names, index=default, format_func=labels.get,
                             key="forecast_model")

try:
    forecast = models[choice].fit(series).predict(range(1, FORECAST_DAYS + 1))
except InsufficientDataError as exc:
    st.info(f"Este modelo não pode prever com os dados atuais: {exc}")
    st.stop()

fc = forecast.to_frame()
measured = (
    summary[summary["model"] == choice].set_index("horizon")["coverage"]
    if not summary.empty else pd.Series(dtype=float)
)
tiles = []
for h in (7, 14, 30):
    row = fc[fc["horizon_days"] == h].iloc[0]
    tiles.append((
        f"Em {h} dias · {fc.index[h - 1].strftime('%d/%m')}", f"{row['mean']:.1f} kg",
        f"IC95% {row['lower']:.1f} – {row['upper']:.1f} kg",
    ))
if 30 in measured.index:
    tiles.append(("Cobertura medida · 30 d", f"{measured[30]:.0%}",
                  "do intervalo, no seu histórico"))
ui.stats(tiles, accent=2)

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
theme.style_figure(fig, height=420, range_slider=False)
st.plotly_chart(fig, width="stretch", config=theme.PLOTLY_CONFIG)

if measured.empty:
    st.caption(
        "Previsão é estimativa: a faixa é o intervalo de 95% que as premissas do modelo "
        "implicam. Ainda não há histórico para medir quanto ela acerta na prática."
    )
else:
    st.caption(
        "A faixa é o intervalo de 95% que as premissas do modelo implicam. No backtesting deste "
        "histórico, a cobertura real foi: "
        + ", ".join(f"{h} d → {c:.0%}" for h, c in measured.items())
        + ". **Recomendado** por padrão: no benchmark com 30 séries sintéticas "
        "(docs/benchmark.md), a combinação Kalman + Holt foi a única calibrada em todos os "
        "horizontes e nunca a pior em nenhum regime."
    )

# --- evaluation ----------------------------------------------------------------------------
st.subheader("Avaliação dos modelos (backtesting)")
if summary.empty:
    st.info(
        "Ainda não há histórico suficiente para avaliar os modelos: é preciso pelo menos "
        f"{DEFAULT_MIN_HISTORY_DAYS} dias de registros e medições reais nas datas-alvo "
        f"(horizontes de {', '.join(map(str, DEFAULT_HORIZONS))} dias)."
    )
    st.stop()

# One row per model: mean absolute error per horizon + calibration at the longest one (the full
# per-horizon table is 28 rows - kept in an expander). Sorted by the mean over horizons, the same
# criterion as the "menor erro" label.
longest = max(summary["horizon"])
mae = summary.pivot(index="model", columns="horizon", values="mae")
compact = pd.DataFrame({
    "Modelo": list(mae.index),
    **{f"Erro {h}d": mae[h].to_numpy() for h in mae.columns},
    # in % and shown without decimals: "97,62%" from ~40 cases is false precision
    f"Cobertura {longest}d": 100 * summary[summary["horizon"] == longest]
    .set_index("model")["coverage"].reindex(mae.index).to_numpy(),
}).iloc[mae.mean(axis=1).to_numpy().argsort()]
badges = [ui.pill(f"Recomendado: {recommended}")]
if best is not None and best != recommended:
    badges.append(ui.pill(f"Menor erro médio aqui: {best}"))
st.markdown("".join(badges), unsafe_allow_html=True)
st.dataframe(
    compact,
    hide_index=True,
    width="stretch",
    column_config={
        "Modelo": st.column_config.TextColumn(width="medium"),
        **{c: st.column_config.NumberColumn(format="%.2f", help="Erro absoluto médio (kg)")
           for c in compact.columns if c.startswith("Erro")},
        f"Cobertura {longest}d": st.column_config.NumberColumn(format="%.0f%%"),
    },
)
st.caption(
    f"Para cada data passada (a cada {BACKTEST_STEP_DAYS} dias), cada modelo é ajustado só com os "
    "dados até aquele dia e comparado com a medição real N dias depois (sem interpolar dias "
    "faltantes), todos nos mesmos casos. **Erro** = erro absoluto médio, em kg. **Cobertura** = fração "
    "de vezes que o valor real caiu dentro do intervalo de 95%: bem abaixo de 95% = confiante "
    "demais; 100% com intervalo largo = cauteloso a ponto de ser pouco útil."
)

if not significance.empty:
    verdicts = significance.apply(_verdict, axis=1)
    shown = verdicts == "diferença demonstrada"
    if shown.any():
        pairs = significance[shown]
        text = "; ".join(f"{r.best} vs. {r.other} em {r.horizon} d"
                         for r in pairs.itertuples())
        ui.callout("Diferença demonstrada (p < 0,01)", text)
    else:
        st.info(
            "Nenhuma diferença de erro entre os modelos foi *demonstrada* neste histórico — "
            "a ordem da tabela pode ser acaso. Por isso o padrão é o modelo calibrado no "
            "benchmark, não o de menor erro aqui."
        )

with st.expander("Todas as métricas por horizonte"):
    table = summary.assign(
        coverage=100 * summary["coverage"], skill_vs_ref=100 * summary["skill_vs_ref"]
    ).rename(
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
        width="stretch",
        column_config={
            "MAE (kg)": st.column_config.NumberColumn(format="%.2f"),
            "RMSE (kg)": st.column_config.NumberColumn(format="%.2f"),
            "Viés (kg)": st.column_config.NumberColumn(format="%+.2f"),
            "Cobertura IC95%": st.column_config.NumberColumn(format="%.0f%%"),
            "Largura média IC (kg)": st.column_config.NumberColumn(format="%.2f"),
            f"Ganho vs. {REFERENCE}": st.column_config.NumberColumn(format="%+.0f%%"),
        },
    )
    st.caption("**Viés** positivo = o peso real ficou acima do previsto.")

if not significance.empty:
    with st.expander("O menor erro é de fato menor? (teste de Diebold-Mariano)"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Horizonte (dias)": significance["horizon"],
                    "Menor MAE": significance["best"],
                    "Comparado com": significance["other"],
                    "Casos independentes": significance["effective_cases"],
                    "p (Holm)": significance["p_holm"],
                    "Conclusão": verdicts,
                }
            ),
            hide_index=True,
            width="stretch",
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
