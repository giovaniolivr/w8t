from datetime import date

import plotly.graph_objects as go
import streamlit as st

from w8t.app import theme, ui
from w8t.core import metrics
from w8t.data import periods, repository
from w8t.data.db import get_session
from w8t.data.models import GoalDirection

ui.setup("Períodos")
ui.header(
    "Períodos",
    "Cada fase com seu objetivo — perda, ganho ou manutenção. Todo registro de peso pertence a "
    "um período, e as análises respeitam essas fronteiras.",
)

DIRECTION_LABELS = {
    GoalDirection.LOSS: "Perda de peso",
    GoalDirection.GAIN: "Ganho de peso",
    GoalDirection.MAINTENANCE: "Manutenção",
}
DIRECTION_BY_LABEL = {v: k for k, v in DIRECTION_LABELS.items()}


def _fmt(d: date) -> str:
    return d.strftime("%d/%m/%Y")


if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

# --- entries that belong to no period ---------------------------------------------------------
with get_session() as session:
    loose = periods.uncovered_entries(session)
    candidates = []
    for p in periods.list_periods(session):
        start, end = periods.attachable_range(session, p)
        covered = [e for e in loose if start <= e.entry_date <= (end or date.max)]
        if covered:
            candidates.append((p, start, end, len(covered)))

if loose:
    ui.callout(
        f"{len(loose)} registro(s) sem período",
        f"De {_fmt(loose[0].entry_date)} a {_fmt(loose[-1].entry_date)}. Vincule-os a um período "
        "existente (as datas dele são estendidas para incluí-los) ou crie um período para eles.",
    )
    for p, start, end, n in candidates:
        span = f"{_fmt(start)} → {_fmt(end) if end else 'em andamento'}"
        if st.button(f"Vincular {n} registro(s) a '{p.label}' ({span})", key=f"attach_{p.id}",
                     type="primary"):
            with get_session() as session:
                try:
                    attached = periods.attach_uncovered(session, p.id)
                    st.session_state["flash"] = f"{attached} registro(s) vinculados a '{p.label}'."
                except periods.OverlappingPeriodError as exc:
                    st.session_state["flash"] = str(exc)
            st.rerun()
    if not candidates:
        st.caption("Nenhum período vizinho pode ser estendido sem sobrepor outro — crie um abaixo.")
    if st.button("Criar período para esses registros", key="prefill_loose"):
        st.session_state["new_period_start"] = loose[0].entry_date  # fixed-key widget state
        st.rerun()

st.subheader("Novo período")

with st.form("new_period_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    label = col1.text_input("Nome do período", placeholder="ex.: Cutting verão 2026")
    direction_label = col2.selectbox("Objetivo", list(DIRECTION_LABELS.values()))

    col3, col4, col5 = st.columns(3)
    start_date = col3.date_input("Início", value=date.today(), key="new_period_start")
    ongoing = col4.checkbox("Período em andamento (sem data de fim)", value=True)
    end_date = None if ongoing else col5.date_input("Fim", value=date.today())

    has_target = st.checkbox("Definir meta específica deste período")
    target_weight_kg = (
        st.number_input("Meta (kg)", min_value=20.0, max_value=400.0, step=0.1, format="%.1f")
        if has_target
        else None
    )

    submitted = st.form_submit_button("Salvar", type="primary")

if submitted:
    if not label.strip():
        st.error("Dê um nome ao período.")
    else:
        with get_session() as session:
            try:
                periods.create_period(
                    session,
                    label=label.strip(),
                    goal_direction=DIRECTION_BY_LABEL[direction_label],
                    start_date=start_date,
                    end_date=end_date,
                    target_weight_kg=target_weight_kg,
                )
                st.success(f"Período '{label}' criado.")
            except periods.OverlappingPeriodError as e:
                st.error(str(e))

st.divider()
st.subheader("Períodos cadastrados")

with get_session() as session:
    all_periods = periods.list_periods(session, ascending=False)
    weights = metrics.to_series(
        (e.entry_date, e.weight_kg) for e in repository.list_entries(session)
    )

TODAY = date.today()


def _status(p) -> str:
    if p.end_date is None:
        return "em andamento"
    if p.end_date >= TODAY:
        return f"planejado até {_fmt(p.end_date)}"
    return f"encerrado em {_fmt(p.end_date)}"


def _sparkline(p, s):
    fig = go.Figure(go.Scatter(
        x=s.index, y=s, mode="lines+markers",
        line={"color": theme.GREEN, "width": 2}, marker={"size": 4, "color": theme.GRAY},
    ))
    if p.target_weight_kg is not None:
        fig.add_hline(y=p.target_weight_kg, line_dash="dot", line_color=theme.GRAY)
    # Fixed minimum vertical span: with autoscale a flat maintenance period's 1 kg of scale
    # noise fills the whole card and looks like big swings.
    values = [*s.tolist(), *([p.target_weight_kg] if p.target_weight_kg is not None else [])]
    mid, span = (max(values) + min(values)) / 2, max(max(values) - min(values), 3.0) * 1.1
    fig.update_layout(
        height=90, margin={"l": 0, "r": 0, "t": 4, "b": 0}, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False},
        yaxis={"visible": False, "range": [mid - span / 2, mid + span / 2]},
    )
    st.plotly_chart(fig, width="stretch", config={"staticPlot": True}, key=f"spark_{p.id}")


def _period_card(p) -> None:
    s = metrics.slice_series(weights, p.start_date, p.end_date or TODAY)
    ongoing = p.end_date is None or p.end_date >= TODAY
    with st.container(border=True, key=f"w8t-card-period-{p.id}"):
        st.markdown(
            f"#### {p.label}\n\n{ui.pill(DIRECTION_LABELS[p.goal_direction])}"
            f"{ui.pill(_status(p))}",
            unsafe_allow_html=True,
        )
        st.caption(f"{_fmt(p.start_date)} → {_fmt(p.end_date) if p.end_date else 'hoje'}")
        summary = metrics.summarize(s)
        if summary is None:
            st.caption("Nenhuma medição neste período ainda.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Começo", f"{summary.initial_kg:.1f} kg", _fmt(summary.first_date),
                      delta_color="off", delta_arrow="off")
            c2.metric("Agora" if ongoing else "Final", f"{summary.current_kg:.1f} kg",
                      _fmt(summary.last_date), delta_color="off", delta_arrow="off")
            pace = summary.pace_kg_per_week
            c3.metric("Variação", f"{summary.change_kg:+.1f} kg",
                      f"{pace:+.2f} kg/sem" if pace is not None else f"{summary.n_entries} medições",
                      delta_color="off", delta_arrow="off")
            if len(s) >= 2:
                _sparkline(p, s)
            if p.target_weight_kg is not None:
                progress = metrics.goal_progress(
                    summary.initial_kg, summary.current_kg, p.target_weight_kg
                )
                if progress is not None:
                    if progress < 0:
                        text = f"Meta {p.target_weight_kg:.1f} kg · afastou-se da meta"
                    elif progress >= 1:
                        text = f"Meta {p.target_weight_kg:.1f} kg · alcançada"
                    else:
                        text = f"Meta {p.target_weight_kg:.1f} kg · {progress:.0%} do caminho"
                    st.progress(min(max(progress, 0.0), 1.0), text=text)
        if st.button("Ver análises", key=f"open_{p.id}", icon=":material/insights:"):
            # Dashboard scoped to this period: same charts, trend, plateaus and anomalies.
            st.switch_page("Home.py", query_params={"periodo": str(p.id)})


if not all_periods:
    st.info("Nenhum período cadastrado ainda. Ao registrar o primeiro peso você define o objetivo.")
else:
    columns = st.columns(2)
    for i, p in enumerate(all_periods):
        with columns[i % 2]:
            _period_card(p)

    st.subheader("Editar ou excluir")
    options = {f"{p.label} ({p.start_date.strftime('%d/%m/%Y')})": p.id for p in all_periods}
    selected_label = st.selectbox("Selecione um período", list(options.keys()))
    selected_id = options[selected_label]
    selected = next(p for p in all_periods if p.id == selected_id)

    ec1, ec2 = st.columns(2)
    new_label = ec1.text_input("Nome", value=selected.label, key=f"edit_label_{selected_id}")
    new_direction_label = ec2.selectbox(
        "Objetivo",
        list(DIRECTION_LABELS.values()),
        index=list(DIRECTION_LABELS.values()).index(DIRECTION_LABELS[selected.goal_direction]),
        key=f"edit_direction_{selected_id}",
    )

    ec3, ec4, ec5 = st.columns(3)
    new_start = ec3.date_input(
        "Início", value=selected.start_date, key=f"edit_start_{selected_id}"
    )
    new_ongoing = ec4.checkbox(
        "Em andamento", value=selected.end_date is None, key=f"edit_ongoing_{selected_id}"
    )
    new_end = (
        None
        if new_ongoing
        else ec5.date_input(
            "Fim", value=selected.end_date or date.today(), key=f"edit_end_{selected_id}"
        )
    )

    bc1, bc2 = st.columns(2)
    if bc1.button("Salvar alterações"):
        with get_session() as session:
            try:
                periods.update_period(
                    session,
                    selected_id,
                    label=new_label.strip(),
                    goal_direction=DIRECTION_BY_LABEL[new_direction_label],
                    start_date=new_start,
                    end_date=new_end,
                )
                st.success("Período atualizado.")
                st.rerun()
            except periods.OverlappingPeriodError as e:
                st.error(str(e))
    if bc2.button("Excluir período"):
        with get_session() as session:
            periods.delete_period(session, selected_id)
        st.success("Período excluído.")
        st.rerun()
