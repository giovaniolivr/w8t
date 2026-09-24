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
scope = options[st.selectbox("Escopo", list(options))]

today = date.today()
summary = build_summary(series, all_periods, scope, today)

with st.expander("Números enviados ao modelo de linguagem (única fonte do texto)"):
    st.json(summary)

if settings.is_demo:
    if scope is not None:
        st.info("Na demonstração, o texto está disponível para o histórico completo.")
        st.stop()
    st.markdown(narrator.DEMO_TEXT_PATH.read_text(encoding="utf-8"))
    st.caption(
        "Texto fixo, gerado uma única vez (Gemini) a partir dos dados de demonstração originais "
        "— a demo pública não chama a API. Se os dados foram editados, ele pode não "
        "corresponder; use o botão de restaurar na página inicial."
    )
    st.stop()

if not settings.gemini_api_key:
    st.info(
        "Para gerar o texto, configure `GEMINI_API_KEY` no arquivo `.env` (chave gratuita do "
        "Google AI Studio). Os números acima continuam disponíveis sem ela."
    )
    st.stop()

key = repr(summary)
if st.button("Gerar resumo", type="primary"):
    try:
        provider = providers.GeminiProvider(settings.gemini_api_key, settings.gemini_model)
        st.session_state["narrative"] = (key, narrator.narrate(summary, provider))
    except providers.ProviderError as exc:
        st.session_state["narrative"] = (key, str(exc))

stored = st.session_state.get("narrative")
if stored is None or stored[0] != key:
    st.caption("O texto só é gerado quando você clicar — cada clique usa a cota gratuita.")
    st.stop()

result = stored[1]
if isinstance(result, str):
    st.error(f"Não foi possível gerar o texto: {result}")
    st.stop()

st.markdown(result.text)
if result.unverified_numbers:
    st.warning(
        "O texto contém números que não estão entre os calculados: "
        + ", ".join(result.unverified_numbers)
        + ". Confira-os com os números acima antes de confiar neles."
    )
else:
    st.caption(
        f"Gerado por {result.provider}. Todos os números do texto conferem com os números "
        "calculados acima."
    )
