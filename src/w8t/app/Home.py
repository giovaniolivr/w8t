from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from w8t.app import theme
from w8t.config import settings
from w8t.core import anomaly, metrics, plateau, trend
from w8t.data import demo, repository
from w8t.data import periods as periods_repo
from w8t.data.db import get_session
from w8t.data.models import GoalDirection

GOAL_LABELS = {
    GoalDirection.LOSS: "perda",
    GoalDirection.GAIN: "ganho",
    GoalDirection.MAINTENANCE: "manutenção",
}
FULL_HISTORY = "Histórico completo"

st.set_page_config(page_title="W8T", page_icon=":chart_with_downwards_trend:", layout="wide")

st.title("W8T")
st.caption(
    f"Acompanhamento inteligente de peso — estatística e Machine Learning · modo "
    f"**{settings.app_env}**"
)

if settings.is_demo:
    st.warning(
        "**Modo demonstração** — todos os dados exibidos são sintéticos, gerados para ilustrar "
        "o app. Qualquer visitante pode editá-los; use o botão na barra lateral para restaurar."
    )
    if st.sidebar.button("Resetar dados de demonstração", type="primary"):
        with get_session() as session:
            demo.reset_demo_data(session, today=date.today())
        st.rerun()

with get_session() as session:
    points = [(e.entry_date, e.weight_kg) for e in repository.list_entries(session)]
    all_periods = periods_repo.list_periods(session)

full_series = metrics.to_series(points)

if full_series.empty:
    st.info(
        "Nenhum registro ainda. Use **Resetar dados de demonstração** na barra lateral."
        if settings.is_demo
        else "Nenhum registro de peso ainda. Comece pela página **Registro de Peso** no menu "
        "lateral."
    )
    st.stop()


def _period_option(p) -> str:
    end = p.end_date.strftime("%d/%m/%Y") if p.end_date else "em andamento"
    return f"{p.label} ({GOAL_LABELS[p.goal_direction]}, {p.start_date.strftime('%d/%m/%Y')} → {end})"


options = {FULL_HISTORY: None} | {_period_option(p): p for p in reversed(all_periods)}
choice = st.selectbox("Escopo", list(options), help="Analisar o histórico inteiro ou um período.")
period = options[choice]

if period is None:
    series = full_series
else:
    series = metrics.slice_series(full_series, period.start_date, period.end_date or date.today())

summary = metrics.summarize(series)
if summary is None:
    st.info("Nenhum registro de peso dentro deste período.")
    st.stop()


def _kg(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "dados insuficientes"
    return f"{value:+.1f} kg" if signed else f"{value:.1f} kg"


def _d(value: date) -> str:
    return value.strftime("%d/%m/%Y")


c1, c2, c3, c4 = st.columns(4)
c1.metric("Peso atual", _kg(summary.current_kg), f"{summary.change_kg:+.1f} kg no escopo",
          delta_color="off", help=f"Última medição, em {_d(summary.last_date)}.")
c2.metric("Peso inicial", _kg(summary.initial_kg), help=f"Primeira medição, em {_d(summary.first_date)}.")
c3.metric("Mínimo", _kg(summary.min_kg), _d(summary.min_date), delta_color="off", delta_arrow="off")
c4.metric("Máximo", _kg(summary.max_kg), _d(summary.max_date), delta_color="off", delta_arrow="off")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Variação", f"{summary.change_kg:+.1f} kg", f"{summary.change_pct:+.1f}%", delta_color="off")
c6.metric(
    "Ritmo",
    "dados insuficientes" if summary.pace_kg_per_week is None
    else f"{summary.pace_kg_per_week:+.2f} kg/sem",
    help=(
        "Inclinação da reta de mínimos quadrados sobre as medições do escopo. "
        f"Exige ≥ {metrics.PACE_MIN_ENTRIES} medições cobrindo ≥ {metrics.PACE_MIN_SPAN_DAYS} dias."
    ),
)
c7.metric("Média móvel 7 dias", _kg(summary.moving_avg_7d),
          help=f"Exige ≥ {metrics.ROLLING_MIN_OBS[7]} medições nos últimos 7 dias.")
c8.metric("Média móvel 30 dias", _kg(summary.moving_avg_30d),
          help=f"Exige ≥ {metrics.ROLLING_MIN_OBS[30]} medições nos últimos 30 dias.")

if period is not None and period.target_weight_kg is not None:
    remaining = period.target_weight_kg - summary.current_kg
    st.caption(
        f"Meta do período: **{period.target_weight_kg:.1f} kg** · "
        f"diferença até a meta: **{remaining:+.1f} kg** a partir da última medição."
    )

st.caption(
    f"{summary.n_entries} medições em {summary.span_days + 1} dias de calendário."
)



@st.cache_data(show_spinner=False)
def _patterns(scoped: pd.Series):
    # Streamlit reruns the script on every interaction; detection is O(n * window) and only
    # needs recomputing when the scoped series itself changes (cache key = series content).
    return (
        trend.current_trend(scoped),
        plateau.detect_plateaus(scoped),
        anomaly.detect_anomalies(scoped),
    )


current_trend, plateaus, anomalies = _patterns(series)
flagged = anomalies[anomalies["is_anomaly"]]

st.subheader("Padrões")
t1, t2, t3 = st.columns(3)
if current_trend is None:
    t1.metric(f"Tendência ({trend.TREND_WINDOW_DAYS} dias)", "dados insuficientes",
              help=f"Exige ≥ {trend.TREND_MIN_OBS} medições nos últimos "
                   f"{trend.TREND_WINDOW_DAYS} dias do escopo.")
else:
    t1.metric(
        f"Tendência ({trend.TREND_WINDOW_DAYS} dias)",
        current_trend.direction.value,
        f"{current_trend.slope_kg_per_week:+.2f} kg/sem "
        f"(IC95% {current_trend.ci_low_kg_per_week:+.2f} a "
        f"{current_trend.ci_high_kg_per_week:+.2f})",
        delta_color="off",
        delta_arrow="off",  # the direction is the value itself; an arrow could contradict it
        help=(
            "Reta de mínimos quadrados sobre as medições da janela. Só indica direção quando o "
            "intervalo de confiança de 95% exclui zero e a inclinação passa de "
            f"±{trend.STABLE_BAND_KG_PER_WEEK} kg/sem; \"estável\" quando o intervalo inteiro "
            "fica dentro dessa faixa; \"indefinido\" quando o ruído não permite concluir."
        ),
    )
t2.metric("Platôs detectados", len(plateaus),
          help=f"Trechos de ≥ {plateau.PLATEAU_MIN_DAYS} dias em que a inclinação da janela de "
               f"{plateau.PLATEAU_WINDOW_DAYS} dias ficou abaixo de "
               f"±{plateau.PLATEAU_MAX_ABS_SLOPE_KG_PER_WEEK} kg/sem.")
t3.metric("Medições atípicas", len(flagged),
          help="Medições que se desviam da tendência das semanas anteriores além do esperado "
               f"pelo ruído (|z robusto| ≥ {anomaly.ANOMALY_Z_THRESHOLD}). Atípico não significa "
               "errado — a medição é mantida como registrada.")

# Same scoped series as the KPI cards, so the chart's last MA point matches the card value.
ma7 = metrics.rolling_mean(series, 7)
ma30 = metrics.rolling_mean(series, 30)

fig = go.Figure()
fig.add_trace(go.Scatter(x=series.index, y=series, mode="markers", name="Medição real",
                         marker={"size": 7, "color": theme.MEASUREMENT}))
fig.add_trace(go.Scatter(x=ma7.index, y=ma7, mode="lines", name="Média móvel 7d",
                         line={"color": theme.MOVING_AVG_SHORT, "width": 2.5},
                         connectgaps=False))
fig.add_trace(go.Scatter(x=ma30.index, y=ma30, mode="lines", name="Média móvel 30d",
                         line={"color": theme.MOVING_AVG_LONG, "dash": "dash"},
                         connectgaps=False))
for p in plateaus:
    fig.add_vrect(x0=p.start, x1=p.end, fillcolor=theme.PLATEAU_FILL, line_width=0, layer="below",
                  annotation_text="platô", annotation_position="top left")
if not flagged.empty:
    fig.add_trace(go.Scatter(
        x=flagged.index, y=flagged["weight_kg"], mode="markers", name="Medição atípica",
        marker={"size": 13, "color": "rgba(0,0,0,0)",
                "line": {"color": theme.ANOMALY, "width": 2}},
        customdata=flagged[["expected_kg", "robust_z"]],
        hovertemplate="%{y:.1f} kg · esperado %{customdata[0]:.1f} kg · z %{customdata[1]:+.1f}"
                      "<extra></extra>",
    ))
if period is not None and period.target_weight_kg is not None:
    fig.add_hline(y=period.target_weight_kg, line_dash="dot", line_color=theme.TARGET,
                  annotation_text="meta")
fig.update_layout(
    height=420, margin={"l": 10, "r": 10, "t": 30, "b": 10},
    yaxis_title="kg", legend={"orientation": "h", "y": 1.08},
)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Pontos são medições registradas. Linhas são médias derivadas dessas medições — "
    "calculadas apenas quando há medições suficientes na janela; dias sem registro não são "
    "preenchidos. Faixas cinza marcam platôs; círculos âmbar, medições atípicas."
)


def _periods_overlapping(start: date, end: date) -> str:
    hits = [
        f"{p.label} ({GOAL_LABELS[p.goal_direction]})"
        for p in all_periods
        if p.start_date <= end and start <= (p.end_date or date.max)
    ]
    return ", ".join(hits) or "—"


if plateaus:
    st.markdown("**Platôs**")
    st.dataframe(
        pd.DataFrame(
            {
                "Início": [_d(p.start) for p in plateaus],
                "Fim": [_d(p.end) for p in plateaus],
                "Dias": [p.days for p in plateaus],
                "Medições": [p.n_obs for p in plateaus],
                "Peso médio (kg)": [round(p.mean_kg, 1) for p in plateaus],
                "Período(s)": [_periods_overlapping(p.start, p.end) for p in plateaus],
            }
        ),
        hide_index=True,
        column_config={"Peso médio (kg)": st.column_config.NumberColumn(format="%.1f")},
    )
    st.caption(
        "Platô é estabilidade do peso, não um julgamento: em manutenção é o esperado; em perda "
        "ou ganho indica estagnação. O início é aproximado — a detecção usa janelas móveis."
    )

if not flagged.empty:
    st.markdown("**Medições atípicas**")
    st.dataframe(
        pd.DataFrame(
            {
                "Data": [_d(ts.date()) for ts in flagged.index],
                "Peso (kg)": flagged["weight_kg"].round(1).to_list(),
                "Esperado pela tendência (kg)": flagged["expected_kg"].round(1).to_list(),
                "z robusto": flagged["robust_z"].round(1).to_list(),
            }
        ),
        hide_index=True,
        column_config={
            "Peso (kg)": st.column_config.NumberColumn(format="%.1f"),
            "Esperado pela tendência (kg)": st.column_config.NumberColumn(format="%.1f"),
            "z robusto": st.column_config.NumberColumn(format="%+.1f"),
        },
    )
