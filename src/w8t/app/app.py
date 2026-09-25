"""Entry point: explicit navigation (titles with accents, icons, clean ASCII URLs).

Each page stays a standalone script (and is tested standalone with AppTest); this file only
declares them. With ``st.navigation`` the automatic ``pages/`` discovery is off, so a new page
must be registered here.
"""

import sys
from datetime import date
from pathlib import Path

import streamlit as st

HERE = Path(__file__).parent

# Hosted (Streamlit Community Cloud): the dependency installer may not install this project
# itself, only its dependencies - make the src/ layout importable either way.
try:
    import w8t  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(HERE.parents[1]))

# after the path setup above; the config reads Streamlit secrets itself
from w8t.config import SECRETS_STATUS, settings

# Hosted on Streamlit Community Cloud (code under /mount/src) = the public demo. If it isn't in
# demo mode the secrets weren't picked up: say so plainly (key names only) instead of failing
# later with "no such table" on an empty local SQLite file.
if HERE.as_posix().startswith("/mount/src") and not settings.is_demo:
    st.error(
        "**Configuração da demo não encontrada.** Nos *Secrets* do app (Manage app → Settings → "
        "Secrets) deve haver, no nível raiz:\n\n"
        '`APP_ENV = "demo"`  \n`DATABASE_URL = "postgresql://..."`\n\n'
        f"Leitura dos secrets: {'ok' if SECRETS_STATUS['read'] else 'falhou'}"
        f"{' (' + SECRETS_STATUS['error'] + ')' if SECRETS_STATUS['error'] else ''} · "
        f"chaves encontradas: {', '.join(SECRETS_STATUS['keys']) or 'nenhuma'}."
    )
    st.stop()


@st.cache_data(ttl=3600, show_spinner="Preparando os dados de demonstração...")
def _prepare_demo(today: date) -> str | None:
    """Once per hour at most: create/seed/refresh the demo database (see ensure_demo_data)."""
    from w8t.data import demo
    from w8t.data.db import get_session

    with get_session() as session:
        return demo.ensure_demo_data(session, today=today)


if settings.is_demo:
    _prepare_demo(date.today())

PAGES = [
    st.Page(HERE / "Home.py", title="Dashboard", icon=":material/space_dashboard:",
            url_path="", default=True),
    st.Page(HERE / "views" / "1_Registro_de_Peso.py", title="Registro de peso",
            icon=":material/monitor_weight:", url_path="registro"),
    st.Page(HERE / "views" / "2_Periodos.py", title="Períodos",
            icon=":material/date_range:", url_path="periodos"),
    st.Page(HERE / "views" / "3_Previsao.py", title="Previsão",
            icon=":material/insights:", url_path="previsao"),
    st.Page(HERE / "views" / "4_Lacunas.py", title="Lacunas",
            icon=":material/extension:", url_path="lacunas"),
    st.Page(HERE / "views" / "5_Resumo.py", title="Resumo",
            icon=":material/auto_awesome:", url_path="resumo"),
]

st.navigation(PAGES, position="sidebar").run()
