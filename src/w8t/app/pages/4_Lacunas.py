import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from w8t.app import theme
from w8t.core import metrics
from w8t.data import repository
from w8t.data.db import get_session
from w8t.patterns import gaps

CONTEXT_DAYS = 30
METHOD_LABELS = {
    "kalman": "Kalman (recomendado)",
    "gpr": "GPR",
    "linear": "Interpolação linear (referência)",
}

st.set_page_config(page_title="W8T · Lacunas", page_icon=":jigsaw:", layout="wide")
st.title("Lacunas")
st.caption(
    "Dias sem registro ficam sem registro. Se quiser, escolha uma lacuna e peça uma estimativa do "
    "que a balança provavelmente mostraria — ela é calculada na hora, exibida como estimativa e "
    "**nunca é salva** junto com as medições."
)

with get_session() as session:
    series = metrics.to_series(
        (e.entry_date, e.weight_kg) for e in repository.list_entries(session)
    )

found = gaps.find_gaps(series) if len(series) > 1 else []
if not found:
    st.info("Nenhuma lacuna entre registros no histórico.")
    st.stop()


def _label(g: gaps.Gap) -> str:
    span = g.start.strftime("%d/%m/%Y")
    if g.days > 1:
        span += f" a {g.end.strftime('%d/%m/%Y')}"
    return f"{span} ({g.days} {'dia' if g.days == 1 else 'dias'})"


ordered = sorted(found, key=lambda g: g.start, reverse=True)
c1, c2 = st.columns([2, 1])
gap = c1.selectbox(f"Lacuna ({len(found)} no histórico)", ordered, format_func=_label)
method = c2.selectbox("Método", list(METHOD_LABELS), format_func=METHOD_LABELS.get)

key = (gap.start, gap.end, method)
if st.button("Estimar valores da lacuna", type="primary"):
    try:
        st.session_state["reconstruction"] = (key, gaps.reconstruct(series, gap, method))
    except gaps.CannotReconstructError as exc:
        st.session_state["reconstruction"] = (key, str(exc))

stored = st.session_state.get("reconstruction")
if stored is None or stored[0] != key:
    st.stop()
result = stored[1]
if isinstance(result, str):
    st.info(f"Não foi possível estimar esta lacuna: {result}")
    st.stop()

lo = pd.Timestamp(gap.start) - pd.Timedelta(days=CONTEXT_DAYS)
hi = pd.Timestamp(gap.end) + pd.Timedelta(days=CONTEXT_DAYS)
context = series[(series.index >= lo) & (series.index <= hi)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=result.index, y=result["upper"], mode="lines", line={"width": 0},
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=result.index, y=result["lower"], mode="lines", line={"width": 0},
                         fill="tonexty", fillcolor=theme.TREND_STATE_BAND,
                         name="Intervalo 95% da estimativa", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=context.index, y=context, mode="markers", name="Medição real",
                         marker={"size": 7, "color": theme.MEASUREMENT}))
fig.add_trace(go.Scatter(
    x=result.index, y=result["estimate_kg"], mode="markers", name="Estimativa (não é medição)",
    marker={"size": 9, "symbol": "diamond-open", "color": theme.TREND_STATE,
            "line": {"width": 2}},
    hovertemplate="%{y:.1f} kg (estimativa)<extra></extra>",
))
fig.add_vrect(x0=gap.start, x1=gap.end, fillcolor=theme.PLATEAU_FILL, line_width=0,
              layer="below", annotation_text="lacuna", annotation_position="top left")
fig.update_layout(height=380, margin={"l": 10, "r": 10, "t": 30, "b": 10}, yaxis_title="kg",
                  legend={"orientation": "h", "y": 1.1})
st.plotly_chart(fig, width="stretch")

st.dataframe(
    pd.DataFrame(
        {
            "Data": [ts.strftime("%d/%m/%Y") for ts in result.index],
            "Tipo": "estimativa — não é medição",
            "Estimativa (kg)": result["estimate_kg"].to_list(),
            "Intervalo 95% (kg)": [
                f"{a:.1f} – {b:.1f}" for a, b in zip(result["lower"], result["upper"], strict=True)
            ],
        }
    ),
    hide_index=True,
    column_config={"Estimativa (kg)": st.column_config.NumberColumn(format="%.1f")},
)
st.caption(
    "O intervalo é para uma *medição*: inclui a incerteza sobre o peso daquele dia e o ruído "
    "normal da balança. Medições marcadas como atípicas não são usadas como evidência. Em 48 "
    "séries sintéticas com trechos apagados de propósito (docs/gaps_benchmark.md), o Kalman "
    "errou em média 0,1 kg do peso verdadeiro e o intervalo cobriu ~95% das medições apagadas; "
    "a interpolação linear errou mais que o dobro. Padrões semanais (ex.: fim de semana) não são "
    "modelados e pioram todas as estimativas."
)
