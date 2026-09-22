from datetime import date

import pandas as pd
import streamlit as st

from w8t.data import repository
from w8t.data.db import get_session

st.set_page_config(page_title="W8T · Registro de Peso", page_icon=":scales:", layout="wide")
st.title("Registro de Peso")

st.subheader("Novo registro")

with st.form("new_entry_form", clear_on_submit=True):
    col1, col2, col3 = st.columns(3)
    entry_date = col1.date_input("Data", value=date.today(), max_value=date.today())
    weight_kg = col2.number_input(
        "Peso (kg)", min_value=20.0, max_value=400.0, step=0.1, format="%.1f"
    )
    has_time = col3.checkbox("Registrar horário")
    entry_time = st.time_input("Horário") if has_time else None
    submitted = st.form_submit_button("Salvar")

if submitted:
    with get_session() as session:
        try:
            repository.create_entry(
                session, entry_date=entry_date, weight_kg=weight_kg, entry_time=entry_time
            )
            st.success(f"Registro de {entry_date.strftime('%d/%m/%Y')} salvo.")
        except repository.DuplicateEntryError:
            st.session_state["duplicate_date"] = entry_date
            st.session_state["duplicate_weight"] = weight_kg
            st.session_state["duplicate_time"] = entry_time

if "duplicate_date" in st.session_state:
    dup_date = st.session_state["duplicate_date"]
    st.warning(f"Já existe um registro para {dup_date.strftime('%d/%m/%Y')}. O que deseja fazer?")
    c1, c2 = st.columns(2)
    if c1.button("Sobrescrever registro existente"):
        with get_session() as session:
            existing = repository.get_entry_by_date(session, dup_date)
            repository.update_entry(
                session,
                existing.id,
                weight_kg=st.session_state["duplicate_weight"],
                entry_time=st.session_state["duplicate_time"],
            )
        st.success("Registro atualizado.")
        del st.session_state["duplicate_date"]
        del st.session_state["duplicate_weight"]
        del st.session_state["duplicate_time"]
        st.rerun()
    if c2.button("Cancelar"):
        del st.session_state["duplicate_date"]
        del st.session_state["duplicate_weight"]
        del st.session_state["duplicate_time"]
        st.rerun()

st.divider()
st.subheader("Histórico")

with get_session() as session:
    entries = repository.list_entries(session, ascending=False)

if not entries:
    st.info("Nenhum registro ainda. Adicione o primeiro peso acima.")
else:
    rows = [
        {"id": e.id, "Data": e.entry_date, "Peso (kg)": e.weight_kg, "Horário": e.entry_time}
        for e in entries
    ]
    df = pd.DataFrame(rows).set_index("id").sort_values("Data")
    df["Diferença vs. anterior (kg)"] = df["Peso (kg)"].diff().round(2)
    st.dataframe(df.sort_values("Data", ascending=False), use_container_width=True)

    st.subheader("Editar ou excluir")
    options = {f"{e.entry_date.strftime('%d/%m/%Y')} — {e.weight_kg} kg": e.id for e in entries}
    selected_label = st.selectbox("Selecione um registro", list(options.keys()))
    selected_id = options[selected_label]
    current_weight = next(e.weight_kg for e in entries if e.id == selected_id)

    new_weight = st.number_input(
        "Peso (kg)",
        min_value=20.0,
        max_value=400.0,
        step=0.1,
        format="%.1f",
        value=current_weight,
        key=f"edit_weight_{selected_id}",
    )

    ec1, ec2 = st.columns(2)
    if ec1.button("Salvar alterações"):
        with get_session() as session:
            repository.update_entry(session, selected_id, weight_kg=new_weight)
        st.success("Registro atualizado.")
        st.rerun()
    if ec2.button("Excluir registro"):
        with get_session() as session:
            repository.delete_entry(session, selected_id)
        st.success("Registro excluído.")
        st.rerun()
