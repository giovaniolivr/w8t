"""Entry point: explicit navigation (titles with accents, icons, clean ASCII URLs).

Each page stays a standalone script (and is tested standalone with AppTest); this file only
declares them. With ``st.navigation`` the automatic ``pages/`` discovery is off, so a new page
must be registered here.
"""

import contextlib
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

# Root-level Streamlit secrets become environment variables once they're loaded; touch them
# before w8t.config is imported so APP_ENV / DATABASE_URL are there. No secrets file locally.
with contextlib.suppress(Exception):  # FileNotFoundError / StreamlitSecretNotFoundError
    st.secrets.to_dict()

from w8t.config import settings  # after the path/secrets setup above


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
