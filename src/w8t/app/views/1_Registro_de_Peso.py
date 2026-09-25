from datetime import date, timedelta

import pandas as pd
import streamlit as st

from w8t.app import ui
from w8t.data import periods, repository
from w8t.data.db import get_session
from w8t.data.models import GoalDirection

GOALS = {
    GoalDirection.LOSS: "Perder peso",
    GoalDirection.GAIN: "Ganhar peso",
    GoalDirection.MAINTENANCE: "Manter o peso",
}
DEFAULT_LABELS = {
    GoalDirection.LOSS: "Perda de peso",
    GoalDirection.GAIN: "Ganho de peso",
    GoalDirection.MAINTENANCE: "Manutenção",
}

ui.setup("Registro de Peso")
ui.header("Registro de peso", "Um registro por dia. Cada registro pertence a um período com objetivo.")


def _fmt(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def _save_entry(entry_date: date, weight_kg: float, entry_time) -> None:
    with get_session() as session:
        try:
            repository.create_entry(
                session, entry_date=entry_date, weight_kg=weight_kg, entry_time=entry_time
            )
            st.session_state["flash"] = f"Registro de {_fmt(entry_date)} salvo."
        except repository.DuplicateEntryError:
            st.session_state["duplicate"] = (entry_date, weight_kg, entry_time)


with get_session() as session:
    recent = repository.list_entries(session, ascending=False)[:2]
    current_period = periods.get_period_for_date(session, date.today())
    last_30 = sum(
        1 for e in repository.list_entries(session)
        if e.entry_date > date.today() - timedelta(days=30)
    )

if recent:
    last = recent[0]
    diff = (
        f"{last.weight_kg - recent[1].weight_kg:+.1f} kg vs. {_fmt(recent[1].entry_date)}"
        if len(recent) > 1 else "primeiro registro"
    )
    ui.stats([
        ("Último registro", f"{last.weight_kg:.1f} kg", f"{_fmt(last.entry_date)} · {diff}"),
        ("Período atual", current_period.label if current_period else "nenhum",
         GOALS[current_period.goal_direction] if current_period else "defina ao registrar"),
        ("Últimos 30 dias", f"{last_30} de 30", "dias com registro"),
    ])

st.subheader("Novo registro")
with st.form("new_entry_form", clear_on_submit=True):
    col1, col2, col3 = st.columns(3)
    entry_date = col1.date_input(
        "Data", value=date.today(), max_value=date.today(), format="DD/MM/YYYY"
    )
    weight_kg = col2.number_input(
        "Peso (kg)", min_value=20.0, max_value=400.0, step=0.1, format="%.1f",
        # start from the last weight: a daily change is a few hundred grams, not 20 kg
        value=float(recent[0].weight_kg) if recent else 70.0,
    )
    has_time = col3.checkbox("Informar horário")
    entry_time = st.time_input("Horário") if has_time else None
    submitted = st.form_submit_button("Salvar", type="primary")

if submitted:
    with get_session() as session:
        covering = periods.get_period_for_date(session, entry_date)
    if covering is None:
        # Every entry belongs to a period: ask for the goal before saving.
        st.session_state["pending_entry"] = (entry_date, weight_kg, entry_time)
    else:
        _save_entry(entry_date, weight_kg, entry_time)
        if "duplicate" not in st.session_state:
            st.rerun()  # the figures at the top were read before this save

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

# --- the entry's date has no period yet: define one ------------------------------------------
if "pending_entry" in st.session_state:
    p_date, p_weight, p_time = st.session_state["pending_entry"]
    with get_session() as session:
        start = periods.start_including_uncovered(session, p_date)
        loose_before = [
            e for e in periods.uncovered_entries(session) if start <= e.entry_date < p_date
        ]
        last = periods.list_periods(session, ascending=False)
        next_start = min(
            (p.start_date for p in periods.list_periods(session) if p.start_date > p_date),
            default=None,
        )
    ui.callout(
        f"Qual é o objetivo a partir de {_fmt(p_date)}?",
        f"O registro de {p_weight:.1f} kg ainda não pertence a nenhum período. Defina o objetivo "
        "e até quando ele vale — isso dá contexto às análises (platô em manutenção é esperado; "
        "em perda de peso é estagnação).",
    )
    default_goal = last[0].goal_direction if last else GoalDirection.LOSS
    goal = st.radio(
        "Objetivo", list(GOALS), index=list(GOALS).index(default_goal),
        format_func=GOALS.get, horizontal=True, key="pending_goal",
    )
    c1, c2 = st.columns(2)
    label = c1.text_input("Nome do período", value=DEFAULT_LABELS[goal], key=f"pending_label_{goal}")
    indefinite = c2.checkbox("Sem data para terminar", value=next_start is None,
                             key="pending_indefinite")
    end = None
    if not indefinite:
        end = st.date_input(
            "Até quando (planejado)",
            value=next_start - timedelta(days=1) if next_start else p_date + timedelta(days=60),
            min_value=p_date,
            key="pending_end",
            format="DD/MM/YYYY",
        )
    elif next_start is not None:
        st.warning(
            f"Já existe um período começando em {_fmt(next_start)}; este precisa terminar antes."
        )
    if loose_before:
        st.caption(
            f"O período começará em {_fmt(start)} e incluirá {len(loose_before)} registro(s) "
            "anterior(es) que estavam sem período."
        )

    b1, b2 = st.columns([1, 1])
    if b1.button("Criar período e salvar registro", type="primary"):
        with get_session() as session:
            try:
                periods.create_period(
                    session, label=label.strip() or DEFAULT_LABELS[goal], goal_direction=goal,
                    start_date=start, end_date=end,
                )
            except periods.OverlappingPeriodError as exc:
                st.error(str(exc))
                st.stop()
        del st.session_state["pending_entry"]
        _save_entry(p_date, p_weight, p_time)
        st.rerun()
    if b2.button("Cancelar"):
        del st.session_state["pending_entry"]
        st.rerun()

# --- same day already has an entry -----------------------------------------------------------
if "duplicate" in st.session_state:
    dup_date, dup_weight, dup_time = st.session_state["duplicate"]
    st.warning(f"Já existe um registro para {_fmt(dup_date)}. O que deseja fazer?")
    c1, c2 = st.columns(2)
    if c1.button("Sobrescrever registro existente"):
        with get_session() as session:
            existing = repository.get_entry_by_date(session, dup_date)
            repository.update_entry(
                session, existing.id, weight_kg=dup_weight, entry_time=dup_time
            )
        del st.session_state["duplicate"]
        st.session_state["flash"] = "Registro atualizado."
        st.rerun()
    if c2.button("Cancelar", key="cancel_duplicate"):
        del st.session_state["duplicate"]
        st.rerun()

st.divider()
st.subheader("Histórico")

with get_session() as session:
    entries = repository.list_entries(session, ascending=False)
    all_periods = periods.list_periods(session)

if not entries:
    st.info("Nenhum registro ainda. Adicione o primeiro peso acima.")
    st.stop()


def _period_of(d: date) -> str:
    for p in all_periods:
        if p.start_date <= d <= (p.end_date or date.max):
            return p.label
    return "— sem período"


df = pd.DataFrame(
    [
        {"Data": e.entry_date, "Peso (kg)": e.weight_kg,
         "Horário": e.entry_time.strftime("%H:%M") if e.entry_time else "",
         "Período": _period_of(e.entry_date)}
        for e in entries
    ]
).sort_values("Data")
df.insert(2, "Diferença (kg)", df["Peso (kg)"].diff().round(2))
if not df["Horário"].any():
    df = df.drop(columns="Horário")
st.dataframe(
    df.sort_values("Data", ascending=False),
    width="stretch",
    hide_index=True,
    height=min(38 + 35 * len(df), 420),
    column_config={
        "Data": st.column_config.DateColumn(format="DD/MM/YYYY"),
        "Peso (kg)": st.column_config.NumberColumn(format="%.1f"),
        "Diferença (kg)": st.column_config.NumberColumn(
            format="%+.2f", help="Em relação ao registro anterior (que pode ser de dias antes)."
        ),
    },
)
loose = [e for e in entries if _period_of(e.entry_date) == "— sem período"]
if loose:
    st.caption(
        f"{len(loose)} registro(s) sem período — vincule-os na página **Períodos**."
    )

st.subheader("Editar ou excluir")
with st.container(border=True, key="w8t-card-edit-entry"):
    options = {f"{_fmt(e.entry_date)} — {e.weight_kg:.1f} kg": e.id for e in entries}
    ec1, ec2 = st.columns([3, 2])
    selected_label = ec1.selectbox("Registro", list(options.keys()))
    selected_id = options[selected_label]
    current_weight = next(e.weight_kg for e in entries if e.id == selected_id)
    new_weight = ec2.number_input(
        "Peso (kg)", min_value=20.0, max_value=400.0, step=0.1, format="%.1f",
        value=current_weight, key=f"edit_weight_{selected_id}",
    )
    actions = st.container(horizontal=True)
    if actions.button("Salvar alterações", icon=":material/save:"):
        with get_session() as session:
            repository.update_entry(session, selected_id, weight_kg=new_weight)
        st.session_state["flash"] = "Registro atualizado."
        st.rerun()
    if actions.button("Excluir", icon=":material/delete:"):
        st.session_state["confirm_delete"] = selected_id

    # Deleting a measurement can't be undone - ask first.
    if st.session_state.get("confirm_delete") == selected_id:
        st.warning(f"Excluir o registro de {selected_label}? Isso não pode ser desfeito.")
        confirm = st.container(horizontal=True)
        if confirm.button("Sim, excluir", type="primary"):
            with get_session() as session:
                repository.delete_entry(session, selected_id)
            del st.session_state["confirm_delete"]
            st.session_state["flash"] = "Registro excluído."
            st.rerun()
        if confirm.button("Não", key="cancel_delete"):
            del st.session_state["confirm_delete"]
            st.rerun()
