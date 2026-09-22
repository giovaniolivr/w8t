import streamlit as st

from w8t.config import settings

st.set_page_config(page_title="W8T", page_icon=":chart_with_downwards_trend:", layout="wide")

st.title("W8T")
st.caption("Acompanhamento inteligente de peso — estatística e Machine Learning.")

st.info(
    f"Esqueleto do projeto rodando em modo **{settings.app_env}**. "
    "Nenhuma feature foi implementada ainda — este é o ponto de partida."
)
