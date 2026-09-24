from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from w8t.app import theme, ui
from w8t.config import settings
from w8t.core import anomaly, metrics, plateau, trend
from w8t.data import demo, repository
from w8t.data import periods as periods_repo
from w8t.data.db import get_session
from w8t.data.models import GoalDirection
from w8t.patterns import kalman as kal
from w8t.patterns import pipeline, regime

GOAL_LABELS = {
    GoalDirection.LOSS: "perda",
    GoalDirection.GAIN: "ganho",
    GoalDirection.MAINTENANCE: "manutenção",
}
FULL_HISTORY = "Histórico completo"

ui.setup("w8t")
ui.header(
    "Dashboard",
    "Acompanhamento inteligente de peso — estatística e Machine Learning"
    + ("" if not settings.is_demo else " · modo demonstração"),
)

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

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
# Open on the period of the latest entry: periods are the user's regime boundaries, so the
# current phase is the natural default; the full history is one click away.
latest = full_series.index[-1].date()
active = next(
    (p for p in all_periods if p.start_date <= latest <= (p.end_date or date.max)), None
)
default_index = list(options.values()).index(active) if active is not None else 0
choice = st.selectbox(
    "Escopo", list(options), index=default_index,
    help="Por padrão, o período atual. Também dá para ver o histórico inteiro ou outro período.",
)
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


@st.dialog("Começar um novo período")
def _switch_period_dialog(current, suggested):
    st.write(
        f"O período **{current.label}** será encerrado no dia anterior ao início do novo. "
        "Os registros continuam onde estão — só as datas dos períodos mudam."
    )
    goals = list(GOAL_LABELS)
    goal = st.radio("Novo objetivo", goals, index=goals.index(suggested),
                    format_func=lambda g: GOAL_LABELS[g].capitalize(), horizontal=True)
    start = st.date_input("Início do novo período", value=date.today(),
                          min_value=current.start_date + timedelta(days=1), max_value=date.today())
    label = st.text_input("Nome", value=f"{GOAL_LABELS[goal].capitalize()} desde {_d(start)}")
    if st.button("Confirmar", type="primary"):
        with get_session() as session:
            periods_repo.update_period(session, current.id, end_date=start - timedelta(days=1))
            periods_repo.create_period(session, label=label, goal_direction=goal,
                                       start_date=start)
        st.session_state["flash"] = f"Período '{label}' iniciado."
        st.rerun()


c1, c2, c3, c4 = st.columns(4)
c1.metric("Peso atual", _kg(summary.current_kg), f"{summary.change_kg:+.1f} kg no escopo",
          delta_color="off", help=f"Última medição, em {_d(summary.last_date)}.")
c2.metric("Peso inicial", _kg(summary.initial_kg), help=f"Primeira medição, em {_d(summary.first_date)}.")
c3.metric("Mínimo", _kg(summary.min_kg), _d(summary.min_date), delta_color="off", delta_arrow="off")
c4.metric("Máximo", _kg(summary.max_kg), _d(summary.max_date), delta_color="off", delta_arrow="off")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Variação", f"{summary.change_kg:+.1f} kg", f"{summary.change_pct:+.1f}%", delta_color="off")
c6.metric(
    "Ritmo (kg/sem)",
    "dados insuficientes" if summary.pace_kg_per_week is None
    else f"{summary.pace_kg_per_week:+.2f}",
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
def _patterns(scoped: pd.Series) -> pipeline.Patterns:
    # Several model fits; recomputed only when the scoped series changes.
    return pipeline.detect(scoped)


_p = _patterns(series)
method, current_trend, trend_method = _p.method, _p.trend, _p.trend_method
plateaus, anomalies, states = _p.plateaus, _p.anomalies, _p.states
flagged = anomalies[anomalies["is_anomaly"]]

suggestion = regime.suggested_goal(period, current_trend) if period is active else None
if suggestion is not None and st.session_state.get("dismissed_suggestion") != period.id:
    arrow = "subir" if suggestion is GoalDirection.GAIN else "cair"
    ui.callout(
        f"O peso começou a {arrow}",
        f"O período <em>{period.label}</em> tem objetivo de {GOAL_LABELS[period.goal_direction]}, "
        f"mas a tendência recente é de {GOAL_LABELS[suggestion]} "
        f"({current_trend.slope_kg_per_week:+.2f} kg/sem, IC95% "
        f"{current_trend.ci_low_kg_per_week:+.2f} a {current_trend.ci_high_kg_per_week:+.2f}). "
        "Se uma nova fase começou, dá para encerrar este período e iniciar outro — ou ignorar.",
    )
    s1, s2, _ = st.columns([1, 1, 3])
    if s1.button("Iniciar novo período", type="primary"):
        _switch_period_dialog(period, suggestion)
    if s2.button("Ignorar"):
        st.session_state["dismissed_suggestion"] = period.id
        st.rerun()

st.subheader("Padrões")
t1, t2, t3 = st.columns(3)
trend_label = (
    f"Tendência (Kalman, {kal.TREND_WINDOW_DAYS} dias)" if trend_method == "kalman"
    else f"Tendência ({trend.TREND_WINDOW_DAYS} dias)"
)
trend_how = (
    f"Inclinação do nível estimada pelo filtro de Kalman nos últimos {kal.TREND_WINDOW_DAYS} "
    "dias (separa ruído da balança de mudança real de ritmo)."
    if trend_method == "kalman"
    else "Reta de mínimos quadrados sobre as medições da janela (histórico curto demais para "
    "o modelo de Kalman)."
)
if current_trend is None:
    t1.metric(trend_label, "dados insuficientes",
              help=f"Exige ≥ {trend.TREND_MIN_OBS} medições nos últimos "
                   f"{trend.TREND_WINDOW_DAYS} dias do escopo.")
else:
    t1.metric(
        trend_label,
        current_trend.direction.value,
        f"{current_trend.slope_kg_per_week:+.2f} kg/sem "
        f"(IC95% {current_trend.ci_low_kg_per_week:+.2f} a "
        f"{current_trend.ci_high_kg_per_week:+.2f})",
        delta_color="off",
        delta_arrow="off",  # the direction is the value itself; an arrow could contradict it
        help=(
            f"{trend_how} Só indica direção quando o intervalo de confiança de 95% exclui zero "
            f"e a inclinação passa de ±{trend.STABLE_BAND_KG_PER_WEEK} kg/sem; \"estável\" "
            "quando o intervalo inteiro fica dentro dessa faixa; \"indefinido\" quando o ruído "
            "não permite concluir."
        ),
    )
t2.metric(
    "Platôs detectados", len(plateaus),
    help=(
        f"Trechos de ≥ {plateau.PLATEAU_MIN_DAYS} dias em que a inclinação "
        + ("suavizada do Kalman (usa o histórico todo, localiza melhor início e fim)"
           if method == "kalman" else f"da janela de {plateau.PLATEAU_WINDOW_DAYS} dias")
        + f" ficou abaixo de ±{plateau.PLATEAU_MAX_ABS_SLOPE_KG_PER_WEEK} kg/sem."
    ),
)
t3.metric(
    "Medições atípicas", len(flagged),
    help=(
        ("Medições longe do previsto pelo filtro de Kalman com os dias anteriores, além do "
         f"esperado pelo ruído (|z| ≥ {kal.ANOMALY_Z_THRESHOLD}); atípicas são isoladas para não "
         "contaminar os dias seguintes. " if method == "kalman" else
         "Medições que se desviam da tendência das semanas anteriores além do esperado pelo "
         f"ruído (|z robusto| ≥ {anomaly.ANOMALY_Z_THRESHOLD}). ")
        + "Atípico não significa errado — a medição é mantida como registrada."
    ),
)

# Same scoped series as the KPI cards, so the chart's last MA point matches the card value.
ma7 = metrics.rolling_mean(series, 7)
ma30 = metrics.rolling_mean(series, 30)
# With the model trend on screen the moving averages start hidden (one click in the legend).
ma_visible = "legendonly" if states is not None else True

fig = go.Figure()
if states is not None:
    fig.add_trace(go.Scatter(x=states.index, y=states["level_hi"], mode="lines",
                             line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=states.index, y=states["level_lo"], mode="lines",
                             line={"width": 0}, fill="tonexty",
                             fillcolor=theme.TREND_STATE_BAND, name="Faixa 95% da tendência",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=states.index, y=states["level_kg"], mode="lines",
                             name="Tendência estimada (Kalman)",
                             line={"color": theme.TREND_STATE, "width": 2.5},
                             hovertemplate="%{y:.1f} kg (estimativa do modelo)<extra></extra>"))
fig.add_trace(go.Scatter(x=series.index, y=series, mode="markers", name="Medição real",
                         marker={"size": 7, "color": theme.MEASUREMENT}))
fig.add_trace(go.Scatter(x=ma7.index, y=ma7, mode="lines", name="Média móvel 7d",
                         line={"color": theme.MOVING_AVG_SHORT, "width": 2, "dash": "dot"},
                         connectgaps=False, visible=ma_visible))
fig.add_trace(go.Scatter(x=ma30.index, y=ma30, mode="lines", name="Média móvel 30d",
                         line={"color": theme.MOVING_AVG_LONG, "dash": "dash"},
                         connectgaps=False, visible=ma_visible))
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
theme.style_figure(fig, height=480)
st.plotly_chart(fig, width="stretch", config=theme.PLOTLY_CONFIG)
st.caption(
    "Pontos cinza são medições registradas. "
    + ("A linha verde contínua é a tendência estimada pelo modelo (estado suavizado do filtro "
       "de Kalman) com faixa de 95% — em dias sem registro ela é uma estimativa do modelo, não "
       "uma medição. Médias móveis ficam disponíveis na legenda. " if states is not None else
       "Linhas são médias derivadas dessas medições, calculadas apenas quando há medições "
       "suficientes na janela; dias sem registro não são preenchidos. ")
    + "Faixas cinza marcam platôs; círculos âmbar, medições atípicas."
)
if states is not None and states.attrs.get("weekly_range_kg"):
    st.caption(
        f"**Padrão semanal detectado** (variação de ~{states.attrs['weekly_range_kg']:.1f} kg "
        "conforme o dia da semana): a linha de tendência mostra o peso sem esse efeito, e ele é "
        "levado em conta nas previsões, na detecção de atípicas e na estimativa de lacunas."
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
        "ou ganho indica estagnação. "
        + ("Início e fim vêm da inclinação suavizada, que usa o histórico todo — por isso um "
           "platô recente pode ser revisto conforme novos registros chegam."
           if method == "kalman" else
           "O início é aproximado — a detecção usa janelas móveis.")
    )

if not flagged.empty:
    st.markdown("**Medições atípicas**")
    st.dataframe(
        pd.DataFrame(
            {
                "Data": [_d(ts.date()) for ts in flagged.index],
                "Peso (kg)": flagged["weight_kg"].round(1).to_list(),
                "Esperado pela tendência (kg)": flagged["expected_kg"].round(1).to_list(),
                "z (desvio padronizado)": flagged["robust_z"].round(1).to_list(),
            }
        ),
        hide_index=True,
        column_config={
            "Peso (kg)": st.column_config.NumberColumn(format="%.1f"),
            "Esperado pela tendência (kg)": st.column_config.NumberColumn(format="%.1f"),
            "z (desvio padronizado)": st.column_config.NumberColumn(format="%+.1f"),
        },
    )
