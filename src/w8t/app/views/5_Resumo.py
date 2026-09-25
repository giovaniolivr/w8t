from datetime import date

import streamlit as st

from w8t.app import ui
from w8t.config import settings
from w8t.core import metrics
from w8t.data import periods as periods_repo
from w8t.data import repository
from w8t.data.db import get_session
from w8t.insights import narrator, providers
from w8t.insights.summary import GOALS, build_summary

FULL_HISTORY = "Histórico completo"

ui.setup("Resumo")
ui.header(
    "Resumo em linguagem natural",
    "Um modelo de linguagem transforma em texto os números que as outras análises já "
    "calcularam — sem receber suas medições brutas, sem calcular nada e sem sugerir causas.",
)

with get_session() as session:
    series = metrics.to_series(
        (e.entry_date, e.weight_kg) for e in repository.list_entries(session)
    )
    all_periods = periods_repo.list_periods(session)

if series.empty:
    st.info("Nenhum registro de peso ainda.")
    st.stop()


def _option(p) -> str:
    end = p.end_date.strftime("%d/%m/%Y") if p.end_date else "em andamento"
    return f"{p.label} ({GOALS[p.goal_direction]}, {p.start_date.strftime('%d/%m/%Y')} → {end})"


options = {FULL_HISTORY: None} | {_option(p): p for p in reversed(all_periods)}
# Current period by default (same as the dashboard); the demo's fixed text covers the full history.
current = periods_repo.period_containing(all_periods, series.index[-1].date())
default = 0 if settings.is_demo or current is None else list(options.values()).index(current)
scope = options[st.selectbox("Escopo", list(options), index=default)]

today = date.today()
summary = build_summary(series, all_periods, scope, today)



def _key_figures(summary: dict) -> None:
    """The headline numbers of the summary as tiles - the same numbers the text narrates."""
    weight = summary.get("peso") if isinstance(summary.get("peso"), dict) else {}
    tiles = []
    if "atual_kg" in weight:
        tiles.append(("Peso atual", f"{weight['atual_kg']:.1f} kg",
                      f"{weight['variacao_kg']:+.1f} kg desde a 1ª medição"))
    trend = summary.get("tendencia_atual")
    if isinstance(trend, dict):
        lo, hi = trend["intervalo_95_kg_por_semana"]
        tiles.append(("Tendência", trend["direcao"],
                      f"{trend['ritmo_kg_por_semana']:+.2f} kg/sem (IC95% {lo:+.2f} a {hi:+.2f})"))
    else:
        tiles.append(("Tendência", "—", "dados insuficientes"))
    forecast = summary.get("previsao")
    if isinstance(forecast, dict):
        h = forecast["horizontes"][-1]
        lo, hi = h["intervalo_95_kg"]
        tiles.append((f"Previsão · {h['dias_apos_a_ultima_medicao']} dias",
                      f"{h['peso_previsto_kg']:.1f} kg", f"IC95% {lo:.1f} – {hi:.1f} kg"))
    elif isinstance(forecast, str):
        tiles.append(("Previsão", "—", forecast))
    ui.stats(tiles, accent=2 if isinstance(forecast, dict) else None)


def _numbers_expander() -> None:
    with st.expander("Números enviados ao modelo de linguagem (única fonte do texto)"):
        st.json(summary)


def _narrative(text: str, badge: str) -> None:
    with st.container(border=True, key="w8t-card-narrative"):
        st.markdown(ui.pill(badge), unsafe_allow_html=True)
        st.markdown(text)


_key_figures(summary)

if settings.is_demo:
    if scope is not None:
        st.info("Na demonstração, o texto está disponível para o histórico completo.")
        st.stop()
    _narrative(narrator.DEMO_TEXT_PATH.read_text(encoding="utf-8"), "Texto fixo · Gemini")
    _numbers_expander()
    st.caption(
        "Texto fixo, gerado uma única vez (Gemini) a partir dos dados de demonstração originais "
        "— a demo pública não chama a API. Se os dados foram editados, ele pode não "
        "corresponder; use o botão de restaurar na página inicial."
    )
    st.stop()

if not settings.gemini_api_key:
    _numbers_expander()
    st.info(
        "Para gerar o texto, configure `GEMINI_API_KEY` no arquivo `.env` (chave gratuita do "
        "Google AI Studio). Os números acima continuam disponíveis sem ela."
    )
    st.stop()

key = repr(summary)
if st.button("Gerar resumo", type="primary", icon=":material/auto_awesome:"):
    try:
        provider = providers.GeminiProvider(settings.gemini_api_key, settings.gemini_model)
        st.session_state["narrative"] = (key, narrator.narrate(summary, provider))
    except providers.ProviderError as exc:
        st.session_state["narrative"] = (key, str(exc))

stored = st.session_state.get("narrative")
if stored is None or stored[0] != key:
    st.caption("O texto só é gerado quando você clicar — cada clique usa a cota gratuita.")
    _numbers_expander()
    st.stop()

result = stored[1]
if isinstance(result, str):
    st.error(f"Não foi possível gerar o texto: {result}")
    st.stop()

verified = not result.unverified_numbers
_narrative(result.text, "Números verificados" if verified else "Números a conferir")
if not verified:
    st.warning(
        "O texto contém números que não estão entre os calculados: "
        + ", ".join(result.unverified_numbers)
        + ". Confira-os com os números enviados antes de confiar neles."
    )
else:
    st.caption(
        f"Gerado por {result.provider}. Todos os números do texto conferem com os números "
        "calculados."
    )
_numbers_expander()
