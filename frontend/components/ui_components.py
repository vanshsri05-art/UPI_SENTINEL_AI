"""Reusable UI building blocks: theme CSS, KPI cards, badges, Plotly styling, tables."""
from __future__ import annotations

import html as html_lib
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

BLUE, CYAN, GREEN, AMBER, RED, PURPLE = "#2E7BFF", "#00C2FF", "#22C55E", "#F59E0B", "#EF4444", "#8B5CF6"
TONES = {"blue": BLUE, "cyan": CYAN, "green": GREEN, "amber": AMBER, "red": RED, "purple": PURPLE}
RISK_COLORS = {"Low": GREEN, "Medium": AMBER, "High": RED}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'Inter', 'Segoe UI', sans-serif !important; }
.stApp { background: radial-gradient(1200px 600px at 85% -10%, rgba(46,123,255,.16), transparent 60%), radial-gradient(900px 500px at -10% 110%, rgba(0,194,255,.08), transparent 60%), #0A1128; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebarNav"] { display: none; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0B1636 0%, #08102A 100%); border-right: 1px solid #1B2A52; }
.brand { display:flex; align-items:center; gap:.7rem; padding:.4rem .2rem 1rem .2rem; }
.brand-logo { width:42px; height:42px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:1.4rem; background:linear-gradient(135deg,#2E7BFF,#00C2FF); box-shadow:0 6px 20px rgba(46,123,255,.45); }
.brand-name { font-weight:800; letter-spacing:.08em; font-size:1.05rem; color:#F2F6FF; line-height:1.1; }
.brand-sub { font-size:.66rem; color:#8EA2CC; line-height:1.25; margin-top:2px; }
.user-chip { background:#0F1A3A; border:1px solid #1E2D55; border-radius:12px; padding:.6rem .8rem; margin:.4rem 0 .8rem 0; font-size:.82rem; color:#C9D6F2; }
.status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#22C55E; margin-right:6px; animation:pulse 2s infinite; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(34,197,94,.6);} 70%{box-shadow:0 0 0 8px rgba(34,197,94,0);} 100%{box-shadow:0 0 0 0 rgba(34,197,94,0);} }
@keyframes fadeUp { from{opacity:0; transform:translateY(12px);} to{opacity:1; transform:translateY(0);} }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap:.25rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label { background:transparent; border:1px solid transparent; border-radius:12px; padding:.62rem .8rem; transition:all .2s ease; width:100%; cursor:pointer; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background:#10204A; border-color:#223564; transform:translateX(3px); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) { background:linear-gradient(90deg, rgba(46,123,255,.28), rgba(0,194,255,.08)); border-color:#2E7BFF; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display:none; }
section[data-testid="stSidebar"] div[role="radiogroup"] p { font-weight:600; font-size:.92rem; color:#DCE6FF; }
.hero { padding:1.2rem 1.5rem; border-radius:18px; margin-bottom:1.2rem; border:1px solid #1E2D55; background:linear-gradient(120deg, rgba(46,123,255,.20), rgba(0,194,255,.05) 60%, rgba(15,26,58,.6)); animation:fadeUp .5s ease both; }
.hero h1 { margin:0; font-size:1.7rem; font-weight:800; color:#F2F6FF; letter-spacing:-.01em; }
.hero p { margin:.25rem 0 0 0; color:#9DB0D8; font-size:.92rem; }
.kpi { background:linear-gradient(160deg,#111C3A,#0E1832); border:1px solid #1E2D55; border-radius:16px; padding:1rem 1.1rem; margin-bottom:.9rem; position:relative; overflow:hidden; transition:transform .25s ease, border-color .25s ease, box-shadow .25s ease; animation:fadeUp .55s ease both; }
.kpi:hover { transform:translateY(-4px); border-color:var(--accent); box-shadow:0 10px 30px rgba(0,0,0,.35), 0 0 0 1px var(--accent) inset; }
.kpi::after { content:""; position:absolute; right:-30px; top:-30px; width:110px; height:110px; border-radius:50%; background:var(--accent); opacity:.10; }
.kpi-top { display:flex; justify-content:space-between; align-items:center; }
.kpi-label { font-size:.74rem; text-transform:uppercase; letter-spacing:.09em; color:#8EA2CC; font-weight:600; }
.kpi-icon { width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1.1rem; background:rgba(255,255,255,.06); }
.kpi-value { font-size:1.75rem; font-weight:800; color:#F2F6FF; margin-top:.35rem; letter-spacing:-.02em; }
.kpi-hint { font-size:.74rem; color:#7F93BE; margin-top:.15rem; }
.panel { background:#0F1A3A; border:1px solid #1E2D55; border-radius:16px; padding:1rem 1.2rem; margin-bottom:1rem; animation:fadeUp .6s ease both; }
.panel h4 { margin:0 0 .4rem 0; color:#EAF0FF; font-size:1rem; }
.section-title { font-size:1.05rem; font-weight:700; color:#EAF0FF; margin:1.1rem 0 .5rem 0; display:flex; align-items:center; gap:.5rem; }
.section-title::before { content:""; width:4px; height:18px; border-radius:2px; background:linear-gradient(#2E7BFF,#00C2FF); }
.badge { display:inline-block; padding:.2rem .65rem; border-radius:999px; font-size:.75rem; font-weight:700; letter-spacing:.03em; }
.verdict { border-radius:18px; padding:1.3rem 1.5rem; border:1px solid; animation:fadeUp .5s ease both; }
.verdict .v-title { font-size:1.9rem; font-weight:800; letter-spacing:-.02em; margin:0; }
.verdict .v-sub { color:#C9D6F2; font-size:.9rem; margin-top:.2rem; }
.callout { border-left:4px solid var(--c); background:rgba(255,255,255,.03); border-radius:10px; padding:.8rem 1rem; margin:.6rem 0; color:#D5E0FA; font-size:.92rem; }
.factor { display:flex; gap:.6rem; align-items:flex-start; padding:.55rem .2rem; border-bottom:1px solid #182548; font-size:.9rem; color:#D5E0FA; }
.factor:last-child { border-bottom:none; }
.login-wrap { max-width:460px; margin:6vh auto 0 auto; }
.login-card { background:linear-gradient(160deg,#111C3A,#0C1530); border:1px solid #223564; border-radius:22px; padding:1.8rem 1.8rem .6rem 1.8rem; box-shadow:0 20px 60px rgba(0,0,0,.45); animation:fadeUp .6s ease both; }
.stButton > button, [data-testid="stFormSubmitButton"] button, .stDownloadButton button { width:100%; border-radius:12px; font-weight:700; letter-spacing:.03em; border:1px solid #2E7BFF; background:linear-gradient(135deg,#2E7BFF,#1D5FE0); color:white; padding:.65rem 1rem; transition:all .2s ease; }
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover, .stDownloadButton button:hover { transform:translateY(-2px); box-shadow:0 8px 24px rgba(46,123,255,.45); border-color:#00C2FF; color:#fff; }
[data-testid="stFormSubmitButton"] button { font-size:1.02rem; padding:.85rem 1rem; background:linear-gradient(135deg,#00A3FF,#2E7BFF 55%,#5B5BFF); }
div[data-baseweb="input"], div[data-baseweb="select"] > div { background:#0C1632 !important; border-radius:10px !important; border-color:#223564 !important; }
.stTabs [data-baseweb="tab-list"] { gap:.4rem; border-bottom:1px solid #1B2A52; }
.stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; padding:.55rem 1rem; color:#9DB0D8; font-weight:600; }
.stTabs [aria-selected="true"] { color:#fff; background:rgba(46,123,255,.15); }
[data-testid="stDataFrame"] { border:1px solid #1E2D55; border-radius:12px; overflow:hidden; }
[data-testid="stForm"] { background:#0F1A3A; border:1px solid #1E2D55; border-radius:18px; padding:1.3rem; }
@media (max-width: 768px) { .hero h1 { font-size:1.3rem; } .kpi-value { font-size:1.4rem; } .block-container { padding-left:.8rem; padding-right:.8rem; } .verdict .v-title { font-size:1.4rem; } }
</style>
"""


def inject_css() -> None:
    st.markdown("\n".join(l for l in _CSS.splitlines() if l.strip()), unsafe_allow_html=True)


def _html(markup: str) -> None:
    """Render HTML without markdown mangling (no indentation / blank lines)."""
    st.markdown(" ".join(line.strip() for line in markup.splitlines() if line.strip()), unsafe_allow_html=True)


def esc(value) -> str:
    return html_lib.escape(str(value))


# ------------------------------------------------------------------ layout pieces
def page_header(title: str, subtitle: str, icon: str = "") -> None:
    _html(f'<div class="hero"><h1>{esc(icon)} {esc(title)}</h1><p>{esc(subtitle)}</p></div>')


def section(title: str) -> None:
    _html(f'<div class="section-title">{esc(title)}</div>')


def kpi_card(label: str, value: str, icon: str, tone: str = "blue", hint: str = "") -> None:
    color = TONES.get(tone, BLUE)
    _html(f'<div class="kpi" style="--accent:{color}"><div class="kpi-top"><span class="kpi-label">{esc(label)}</span>'
          f'<span class="kpi-icon">{icon}</span></div><div class="kpi-value">{esc(value)}</div>'
          f'<div class="kpi-hint">{esc(hint) if hint else "&nbsp;"}</div></div>')


def kpi_row(cards: List[Dict]) -> None:
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            kpi_card(**card)


def badge(text: str, color: str) -> str:
    return (f'<span class="badge" style="background:{color}22;color:{color};border:1px solid {color}66">'
            f'{esc(text)}</span>')


def risk_badge(level: str) -> str:
    return badge(f"{level.upper()} RISK", RISK_COLORS.get(level, BLUE))


def callout(text: str, tone: str = "blue") -> None:
    _html(f'<div class="callout" style="--c:{TONES.get(tone, BLUE)}">{esc(text)}</div>')


def empty_state(message: str) -> None:
    _html(f'<div class="panel" style="text-align:center;padding:2.2rem"><div style="font-size:2rem">🛰️</div>'
          f'<h4>No data yet</h4><p style="color:#9DB0D8;margin:0">{esc(message)}</p></div>')


def api_error(err) -> None:
    st.error(f"{err.detail}" + (f" (HTTP {err.status})" if err.status else ""))


def inr(value: float) -> str:
    value = float(value)
    if abs(value) >= 1e7:
        return f"\u20b9{value / 1e7:.2f} Cr"
    if abs(value) >= 1e5:
        return f"\u20b9{value / 1e5:.2f} L"
    return f"\u20b9{value:,.0f}"


# ------------------------------------------------------------------ charts
def style_fig(fig: go.Figure, height: int = 340, title: Optional[str] = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#C9D6F2", size=12),
        title=dict(text=title, x=0.01, font=dict(size=15, color="#EAF0FF")) if title else None,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10), showlegend=legend,
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0F1A3A", bordercolor="#2E7BFF", font_size=12),
        colorway=[BLUE, CYAN, GREEN, AMBER, RED, PURPLE])
    fig.update_xaxes(gridcolor="rgba(255,255,255,.05)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)", zeroline=False)
    return fig


def show_chart(fig: go.Figure, key: str) -> None:
    config = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
    try:
        st.plotly_chart(fig, key=key, config=config, width="stretch")
    except Exception:  # older Streamlit versions
        st.plotly_chart(fig, key=key, config=config, use_container_width=True)


def show_table(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def dual_axis(x, bar_y, line_y, bar_name: str, line_name: str, height: int = 340, title: str = "",
              bar_color: str = BLUE, line_color: str = RED) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=x, y=bar_y, name=bar_name, marker_color=bar_color, opacity=0.85), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=line_y, name=line_name, mode="lines+markers",
                             line=dict(color=line_color, width=3), marker=dict(size=6)), secondary_y=True)
    style_fig(fig, height, title)
    fig.update_yaxes(title_text=bar_name, secondary_y=False)
    fig.update_yaxes(title_text=line_name, secondary_y=True, showgrid=False)
    return fig


def risk_gauge(score: float, low_max: int, medium_max: int, height: int = 280) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(font=dict(size=54, color="#F2F6FF")),
        gauge=dict(axis=dict(range=[0, 100], tickcolor="#6C80AD"), bar=dict(color="#EAF0FF", thickness=0.22),
                   bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[dict(range=[0, low_max], color="rgba(34,197,94,.55)"),
                          dict(range=[low_max, medium_max], color="rgba(245,158,11,.55)"),
                          dict(range=[medium_max, 100], color="rgba(239,68,68,.6)")])))
    return style_fig(fig, height, legend=False)


def factor_chart(factors: List[Dict], height: int = 320) -> go.Figure:
    data = list(reversed(factors))
    fig = go.Figure(go.Bar(
        x=[f["relative_impact"] for f in data], y=[f["label"] for f in data], orientation="h",
        marker_color=[RED if f["contribution"] > 0 else BLUE for f in data],
        text=[f"{f['relative_impact']:+.1f}%" for f in data], textposition="outside",
        hovertext=[f"{f['label']}: {f['value']}" for f in data], hoverinfo="text"))
    style_fig(fig, height, legend=False)
    fig.update_xaxes(title_text="Share of model decision (red = pushes toward fraud, blue = away)")
    return fig


def factor_list(factors: List[Dict]) -> None:
    rows = []
    for f in factors:
        color = RED if f["contribution"] > 0 else BLUE
        arrow = "\u25b2" if f["contribution"] > 0 else "\u25bc"
        rows.append(f'<div class="factor"><span style="color:{color};font-weight:800">{arrow}</span>'
                    f'<span><b>{esc(f["label"])}</b> ({esc(f["value"])}) \u2014 {esc(f["text"])}</span></div>')
    _html('<div class="panel">' + "".join(rows) + "</div>")
