"""Shared page chrome: page config, global CSS (depth, motion), logo and headers.

Every page starts with ``ui.setup(...)``. Styling targets Streamlit's ``data-testid`` hooks;
motion respects ``prefers-reduced-motion``. Palette mirrors ``theme.py`` / .streamlit/config.toml.
"""

from __future__ import annotations

import base64
from pathlib import Path
from urllib.parse import urlsplit

import streamlit as st

STATIC = Path(__file__).parent / "static"
LOGO = STATIC / "logo.svg"
ICON = STATIC / "icon.svg"
_LOGO_B64 = base64.b64encode(LOGO.read_bytes()).decode()
_ICON_B64 = base64.b64encode(ICON.read_bytes()).decode()

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --w8t-green: #22C55E;
  --w8t-green-soft: rgba(34, 197, 94, 0.14);
  --w8t-bg: #0A0A0A;
  --w8t-surface: #151817;
  --w8t-surface-2: #1F2322;
  --w8t-border: rgba(255, 255, 255, 0.07);
  --w8t-text: #E5E7EB;
  --w8t-muted: #9CA3AF;
  --w8t-shadow: 0 10px 30px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

/* Text only - never icons: Streamlit's icons are a font too (Material Symbols), and overriding
   every element's font turns them into their ligature names (e.g. "keyboard_double..."). */
html, body, .stApp, .stMarkdown, p, li, label, h1, h2, h3, h4, h5, h6,
button, input, textarea, [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] {
  font-family: 'Material Symbols Rounded' !important;
}

/* Depth: a faint green glow behind the content, darker toward the edges. */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 500px at 15% -10%, rgba(34, 197, 94, 0.10), transparent 60%),
    radial-gradient(900px 400px at 110% 10%, rgba(134, 239, 172, 0.05), transparent 60%),
    var(--w8t-bg);
}
[data-testid="stHeader"] { background: transparent; }

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #171b1a 0%, #0d0f0e 100%);
  border-right: 1px solid var(--w8t-border);
}
[data-testid="stSidebarNav"] a { border-radius: 10px; transition: background .2s ease; }
[data-testid="stSidebarNav"] a:hover { background: var(--w8t-green-soft); }

/* Entrance motion */
@keyframes w8t-rise {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes w8t-glow {
  0%, 100% { box-shadow: var(--w8t-shadow), 0 0 0 1px rgba(34, 197, 94, 0.25); }
  50%      { box-shadow: var(--w8t-shadow), 0 0 22px 2px rgba(34, 197, 94, 0.28); }
}
[data-testid="stMainBlockContainer"] { animation: w8t-rise .45s ease-out both; }

/* Page header */
.w8t-hero { margin: 0 0 1.2rem 0; animation: w8t-rise .5s ease-out both; }
.w8t-hero h1 {
  font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; margin: 0;
  background: linear-gradient(90deg, #F3F4F6 0%, #86EFAC 60%, #22C55E 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.w8t-hero p { color: var(--w8t-muted); margin: .35rem 0 0 0; font-size: .95rem; }

/* Section titles with an accent bar */
[data-testid="stMainBlockContainer"] h3 {
  border-left: 3px solid var(--w8t-green); padding-left: .6rem; font-weight: 700;
}

/* Cards: metrics, charts, tables, forms, expanders */
[data-testid="stMetric"],
[data-testid="stPlotlyChart"],
[data-testid="stForm"],
[data-testid="stExpander"] details,
[data-testid="stDataFrame"] {
  background: linear-gradient(150deg, var(--w8t-surface-2) 0%, var(--w8t-surface) 100%);
  border: 1px solid var(--w8t-border);
  border-radius: 16px;
  box-shadow: var(--w8t-shadow);
}
[data-testid="stMetric"] {
  padding: 14px 16px;
  transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
  animation: w8t-rise .5s ease-out both;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-3px);
  border-color: rgba(34, 197, 94, 0.45);
  box-shadow: 0 16px 36px rgba(0, 0, 0, 0.55), 0 0 0 1px rgba(34, 197, 94, 0.18);
}
[data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"] { animation-delay: .06s; }
[data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"] { animation-delay: .12s; }
[data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"] { animation-delay: .18s; }
[data-testid="stMetricLabel"] p {
  text-transform: uppercase; letter-spacing: .06em; font-size: .72rem; color: var(--w8t-muted);
}
[data-testid="stMetricValue"] { font-weight: 700; letter-spacing: -0.01em; }
[data-testid="stPlotlyChart"] { padding: 10px 8px 4px 8px; }
[data-testid="stForm"] { padding: 18px 18px 8px 18px; }
[data-testid="stDataFrame"] { padding: 4px; }

/* Cards built with st.container(border=True, key="w8t-card-...") - the key becomes a stable
   CSS class (st-key-...); in this Streamlit version the bordered block has no wrapper testid. */
[class*="st-key-w8t-card"] {
  background: linear-gradient(150deg, var(--w8t-surface-2) 0%, var(--w8t-surface) 100%);
  border-color: var(--w8t-border) !important;
  border-radius: 18px !important;
  box-shadow: var(--w8t-shadow);
  transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
  animation: w8t-rise .5s ease-out both;
}
[class*="st-key-w8t-card"]:hover {
  border-color: rgba(34, 197, 94, 0.35) !important;
  box-shadow: 0 16px 36px rgba(0, 0, 0, 0.55);
  transform: translateY(-2px);
}
[class*="st-key-w8t-card"] h4 { margin-bottom: .1rem; }
/* Metrics inside a card: lighter and smaller, so three fit side by side in half a row. */
[class*="st-key-w8t-card"] [data-testid="stMetric"] {
  background: rgba(255, 255, 255, 0.025); box-shadow: none; padding: 10px 12px;
}
[class*="st-key-w8t-card"] [data-testid="stMetric"]:hover { transform: none; }
[class*="st-key-w8t-card"] [data-testid="stMetricValue"] { font-size: 1.35rem; }
[class*="st-key-w8t-card"] [data-testid="stMetricValue"] [data-testid="stMarkdownContainer"] {
  overflow: visible; text-overflow: clip;
}
[data-testid="stProgress"] > div > div > div > div { background: linear-gradient(90deg, #16A34A, #86EFAC); }

/* Buttons */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius: 10px; font-weight: 600;
  transition: transform .15s ease, box-shadow .2s ease, filter .2s ease;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 22px rgba(34, 197, 94, 0.22);
}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {
  background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%);
  border: none; color: #04120a;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); }

/* Alerts */
[data-testid="stAlert"] { border-radius: 12px; }

/* Highlighted call-to-action panel (period prompt, regime suggestion) */
.w8t-callout {
  border-radius: 16px; padding: 16px 18px; margin: 4px 0 12px 0;
  background: linear-gradient(135deg, rgba(34, 197, 94, 0.14) 0%, rgba(21, 24, 23, 0.9) 70%);
  border: 1px solid rgba(34, 197, 94, 0.35);
  animation: w8t-rise .4s ease-out both, w8t-glow 3.2s ease-in-out 0.6s 2;
}
.w8t-callout strong { color: #86EFAC; }
.w8t-callout p { margin: .25rem 0 0 0; color: var(--w8t-text); }

/* Small pill (period badge) */
.w8t-pill {
  display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .78rem;
  font-weight: 600; color: #86EFAC; background: var(--w8t-green-soft);
  border: 1px solid rgba(34, 197, 94, 0.3); margin-right: 6px;
}

/* Logo link pinned to the sidebar's header area */
.w8t-logo-link {
  position: fixed; top: 18px; left: 20px; z-index: 999990; display: block;
  transition: transform .2s ease, filter .2s ease;
}
.w8t-logo-link img { height: 32px; display: block; }
.w8t-logo-link .w8t-logo-mini { display: none; }
.w8t-logo-link:hover {
  transform: translateY(-1px) scale(1.04);
  filter: drop-shadow(0 6px 14px rgba(34, 197, 94, .35));
}
[data-testid="stSidebarHeader"] { min-height: 64px; }

/* Collapsed sidebar = slim icon rail instead of disappearing. Streamlit collapses it by sliding
   it -300px and shrinking it to 1px inside a flex row, so undoing that and fixing the width is
   enough - the main area reflows by itself. */
[data-testid="stSidebar"][aria-expanded="false"] {
  transform: none !important;
  width: 72px !important; min-width: 72px !important; max-width: 72px !important;
}
[data-testid="stSidebar"][aria-expanded="false"] > div {
  width: 72px !important; min-width: 72px !important; max-width: 72px !important;
  overflow-x: hidden;
}
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNav"] { padding-top: 64px; }
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavLink"] {
  justify-content: center; padding: 9px 0; margin: 4px 12px;
}
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavLink"]
  [data-testid="stMarkdownContainer"] p { margin: 0; }
/* the "collapse" chevron makes no sense on the rail; the "expand" one stays in the header */
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarHeader"] button {
  display: none;
}
/* Page name as a tooltip on hover. position: fixed with top:auto keeps its natural vertical
   position while escaping the rail's overflow clipping. */
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavLink"]
  [data-testid="stMarkdownContainer"] {
  position: fixed; left: 76px; opacity: 0; pointer-events: none; white-space: nowrap;
  margin-top: -19px;  /* measured: static position sits 19px below the icon's center */
  background: var(--w8t-surface-2); border: 1px solid rgba(34, 197, 94, 0.35);
  border-radius: 8px; padding: 4px 10px; box-shadow: var(--w8t-shadow);
  transition: opacity .15s ease, transform .15s ease; transform: translateX(-4px);
  z-index: 1000000;
}
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavLink"]:hover
  [data-testid="stMarkdownContainer"] { opacity: 1; transform: translateX(0); }
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavLink"]
  [data-testid="stIconMaterial"] { font-size: 1.35rem; }
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarUserContent"] .stButton,
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarUserContent"] hr,
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavSeparator"] {
  display: none;
}
/* ...and the full logo becomes the "8" mark, below the expand button. */
[data-testid="stSidebar"][aria-expanded="false"] .w8t-logo-link { top: 58px; left: 18px; }
[data-testid="stSidebar"][aria-expanded="false"] .w8t-logo-full { display: none; }
[data-testid="stSidebar"][aria-expanded="false"] .w8t-logo-mini { display: block; height: 36px; }

/* Scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #2a2f2d; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #3a403e; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
"""


def _home_url() -> str | None:
    """Absolute URL of the dashboard (st.logo only accepts absolute links)."""
    try:
        url = st.context.url
    except Exception:  # noqa: BLE001 - not available in some runtimes (e.g. tests)
        return None
    if not url:
        return None
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}/" if parts.scheme and parts.netloc else None


def setup(page_title: str, page_icon: str = ":chart_with_downwards_trend:") -> None:
    st.set_page_config(
        page_title=f"w8t · {page_title}" if page_title != "w8t" else "w8t",
        page_icon=str(ICON),
        layout="wide",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    # Logo = navigation back to the dashboard. st.logo's own link always opens a new tab (its
    # click handler, not the target attribute - verified in the browser), so the logo is a plain
    # HTML link with target="_self", pinned to the sidebar's top-left corner via CSS.
    st.sidebar.markdown(
        f'<a class="w8t-logo-link" href="{_home_url() or "./"}" target="_self" '
        f'title="Voltar ao dashboard">'
        f'<img class="w8t-logo-full" src="data:image/svg+xml;base64,{_LOGO_B64}" alt="w8t">'
        f'<img class="w8t-logo-mini" src="data:image/svg+xml;base64,{_ICON_B64}" alt="w8t">'
        "</a>",
        unsafe_allow_html=True,
    )


def header(title: str, subtitle: str | None = None) -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="w8t-hero"><h1>{title}</h1>{sub}</div>', unsafe_allow_html=True)


def callout(title: str, body: str) -> None:
    st.markdown(
        f'<div class="w8t-callout"><strong>{title}</strong><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def pill(text: str) -> str:
    return f'<span class="w8t-pill">{text}</span>'
