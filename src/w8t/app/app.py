"""Entry point: explicit navigation (titles with accents, icons, clean ASCII URLs).

Each page stays a standalone script (and is tested standalone with AppTest); this file only
declares them. With ``st.navigation`` the automatic ``pages/`` discovery is off, so a new page
must be registered here.
"""

from pathlib import Path

import streamlit as st

HERE = Path(__file__).parent

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
