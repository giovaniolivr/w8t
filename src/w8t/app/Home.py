from datetime import date

import plotly.graph_objects as go
import streamlit as st

from w8t.config import settings
from w8t.core import metrics
from w8t.data import periods as periods_repo
from w8t.data import repository
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

with get_session() as session:
    points = [(e.entry_date, e.weight_kg) for e in repository.list_entries(session)]
    all_periods = periods_repo.list_periods(session)

full_series = metrics.to_series(points)

if full_series.empty:
    st.info(
        "Nenhum registro de peso ainda. Comece pela página **Registro de Peso** no menu lateral."
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
c3.metric("Mínimo", _kg(summary.min_kg), _d(summary.min_date), delta_color="off")
c4.metric("Máximo", _kg(summary.max_kg), _d(summary.max_date), delta_color="off")

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
        f"faltam **{remaining:+.1f} kg** em relação à última medição."
    )

st.caption(
    f"{summary.n_entries} medições em {summary.span_days + 1} dias de calendário."
)

# Same scoped series as the KPI cards, so the chart's last MA point matches the card value.
ma7 = metrics.rolling_mean(series, 7)
ma30 = metrics.rolling_mean(series, 30)

fig = go.Figure()
fig.add_trace(go.Scatter(x=series.index, y=series, mode="markers", name="Medição real",
                         marker={"size": 7}))
fig.add_trace(go.Scatter(x=ma7.index, y=ma7, mode="lines", name="Média móvel 7d",
                         connectgaps=False))
fig.add_trace(go.Scatter(x=ma30.index, y=ma30, mode="lines", name="Média móvel 30d",
                         line={"dash": "dash"}, connectgaps=False))
if period is not None and period.target_weight_kg is not None:
    fig.add_hline(y=period.target_weight_kg, line_dash="dot", annotation_text="meta")
fig.update_layout(
    height=420, margin={"l": 10, "r": 10, "t": 30, "b": 10},
    yaxis_title="kg", legend={"orientation": "h", "y": 1.08},
)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Pontos são medições registradas. Linhas são médias derivadas dessas medições — "
    "calculadas apenas quando há medições suficientes na janela; dias sem registro não são "
    "preenchidos."
)
