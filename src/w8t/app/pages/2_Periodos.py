from datetime import date

import pandas as pd
import streamlit as st

from w8t.data import periods
from w8t.data.db import get_session
from w8t.data.models import GoalDirection

st.set_page_config(page_title="W8T · Períodos", page_icon=":dart:", layout="wide")
st.title("Períodos")
st.caption(
    "Opcional: agrupe seu histórico em ciclos com objetivos distintos (cutting, bulk, "
    "manutenção...) para que as análises futuras considerem o contexto de cada fase."
)

DIRECTION_LABELS = {
    GoalDirection.LOSS: "Perda de peso",
    GoalDirection.GAIN: "Ganho de peso",
    GoalDirection.MAINTENANCE: "Manutenção",
}
DIRECTION_BY_LABEL = {v: k for k, v in DIRECTION_LABELS.items()}

st.subheader("Novo período")

with st.form("new_period_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    label = col1.text_input("Nome do período", placeholder="ex.: Cutting verão 2026")
    direction_label = col2.selectbox("Objetivo", list(DIRECTION_LABELS.values()))

    col3, col4, col5 = st.columns(3)
    start_date = col3.date_input("Início", value=date.today())
    ongoing = col4.checkbox("Período em andamento (sem data de fim)", value=True)
    end_date = None if ongoing else col5.date_input("Fim", value=date.today())

    has_target = st.checkbox("Definir meta específica deste período")
    target_weight_kg = (
        st.number_input("Meta (kg)", min_value=20.0, max_value=400.0, step=0.1, format="%.1f")
        if has_target
        else None
    )

    submitted = st.form_submit_button("Salvar")

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

if not all_periods:
    st.info("Nenhum período cadastrado ainda. O histórico continua funcionando normalmente sem eles.")
else:
    rows = [
        {
            "id": p.id,
            "Nome": p.label,
            "Objetivo": DIRECTION_LABELS[p.goal_direction],
            "Início": p.start_date,
            "Fim": p.end_date if p.end_date else "em andamento",
            "Meta (kg)": p.target_weight_kg if p.target_weight_kg else "—",
        }
        for p in all_periods
    ]
    st.dataframe(pd.DataFrame(rows).set_index("id"), use_container_width=True)

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
