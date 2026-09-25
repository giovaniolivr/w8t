import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from w8t.app import theme, ui
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

ui.setup("Lacunas")
ui.header(
    "Lacunas",
    "Dias sem registro ficam sem registro. Se quiser, peça uma estimativa do que a balança "
    "provavelmente mostraria — calculada na hora e nunca salva junto com as medições.",
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


missing_days = sum(g.days for g in found)
span_days = (series.index[-1] - series.index[0]).days + 1
longest = max(found, key=lambda g: (g.days, g.start))
ui.stats([
    ("Lacunas", str(len(found)), "trechos sem registro entre medições"),
    ("Maior lacuna", f"{longest.days} {'dia' if longest.days == 1 else 'dias'}",
     _label(longest).split(" (")[0]),
    ("Dias com registro", f"{1 - missing_days / span_days:.0%}",
     f"{span_days - missing_days} de {span_days} dias"),
])

ordered = sorted(found, key=lambda g: g.start, reverse=True)
with st.container(border=True, key="w8t-card-gap-controls"):
    c1, c2 = st.columns([1, 1])
    gap = c1.selectbox("Lacuna (mais recentes primeiro)", ordered, format_func=_label)
    method = c2.selectbox("Método", list(METHOD_LABELS), format_func=METHOD_LABELS.get)
    key = (gap.start, gap.end, method)
    if st.button("Estimar valores da lacuna", type="primary", icon=":material/auto_fix_high:"):
        try:
            st.session_state["reconstruction"] = (key, gaps.reconstruct(series, gap, method))
        except gaps.CannotReconstructError as exc:
            st.session_state["reconstruction"] = (key, str(exc))

stored = st.session_state.get("reconstruction")
result = stored[1] if stored is not None and stored[0] == key else None
if isinstance(result, str):
    st.info(f"Não foi possível estimar esta lacuna: {result}")
    result = None

# The gap in context is shown right away (measurements only); estimates join the chart only
# after the user asks for them.
lo = pd.Timestamp(gap.start) - pd.Timedelta(days=CONTEXT_DAYS)
hi = pd.Timestamp(gap.end) + pd.Timedelta(days=CONTEXT_DAYS)
context = series[(series.index >= lo) & (series.index <= hi)]

fig = go.Figure()
if result is not None:
    fig.add_trace(go.Scatter(x=result.index, y=result["upper"], mode="lines",
                             line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=result.index, y=result["lower"], mode="lines",
                             line={"width": 0}, fill="tonexty",
                             fillcolor=theme.TREND_STATE_BAND,
                             name="Intervalo 95% da estimativa", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=context.index, y=context, mode="markers", name="Medição real",
                         marker={"size": 7, "color": theme.MEASUREMENT}))
if result is not None:
    fig.add_trace(go.Scatter(
        x=result.index, y=result["estimate_kg"], mode="markers",
        name="Estimativa (não é medição)",
        marker={"size": 9, "symbol": "diamond-open", "color": theme.TREND_STATE,
                "line": {"width": 2}},
        # error bars too: a one-day gap has no band to draw, but still has an interval
        error_y={"type": "data", "symmetric": False,
                 "array": result["upper"] - result["estimate_kg"],
                 "arrayminus": result["estimate_kg"] - result["lower"],
                 "color": theme.TREND_STATE, "thickness": 1.2, "width": 4},
        customdata=result[["lower", "upper"]],
        hovertemplate=("%{y:.1f} kg (estimativa · IC95% %{customdata[0]:.1f}–"
                       "%{customdata[1]:.1f})<extra></extra>"),
    ))
fig.add_vrect(x0=pd.Timestamp(gap.start) - pd.Timedelta(hours=12),
              x1=pd.Timestamp(gap.end) + pd.Timedelta(hours=12),
              fillcolor=theme.PLATEAU_FILL, line_width=0, layer="below",
              annotation_text="lacuna", annotation_position="top left")
theme.style_figure(fig, height=400, range_slider=False)
# legend under the plot: on top it collided with the "lacuna" label and got clipped on the right
fig.update_layout(legend={"y": -0.14, "x": 0, "xanchor": "left", "yanchor": "top"},
                  margin={"b": 70})
st.plotly_chart(fig, width="stretch", config=theme.PLOTLY_CONFIG)

if result is None:
    st.caption(
        "Só medições reais por enquanto. Clique em **Estimar** para ver o que a balança "
        "provavelmente mostraria nos dias sem registro."
    )
    st.stop()

half_width = float(((result["upper"] - result["lower"]) / 2).mean())
ui.stats([
    ("Dias estimados", str(len(result)), METHOD_LABELS[method]),
    ("Estimativa média", f"{result['estimate_kg'].mean():.1f} kg", "não é medição"),
    ("Incerteza típica", f"± {half_width:.1f} kg", "intervalo de 95%"),
], accent=1)

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
    width="stretch",
    column_config={"Estimativa (kg)": st.column_config.NumberColumn(format="%.1f")},
)
st.caption(
    "O intervalo é para uma *medição*: inclui a incerteza sobre o peso daquele dia e o ruído "
    "normal da balança. Medições marcadas como atípicas não são usadas como evidência. Em 48 "
    "séries sintéticas com trechos apagados de propósito (docs/gaps_benchmark.md), o Kalman "
    "errou em média menos de 0,1 kg do peso verdadeiro e o intervalo cobriu ~95% das medições "
    "apagadas; a interpolação linear errou mais que o dobro. Quando há padrão semanal (ex.: fim "
    "de semana), o Kalman o detecta e inclui na estimativa de cada dia."
)
