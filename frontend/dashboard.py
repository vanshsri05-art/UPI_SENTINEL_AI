"""UPI SENTINEL AI - Streamlit dashboard entry point.

    streamlit run frontend/dashboard.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="UPI Sentinel AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

from frontend.api_client import ApiClient, ApiError  # noqa: E402
from frontend.components import ui_components as ui  # noqa: E402
from frontend.pages import (admin_dashboard, fraud_analytics, fraud_detection, overview,  # noqa: E402
                            risk_analysis, transaction_history)

PAGES = {
    "📊  Overview": overview,
    "🛡️  Fraud Detection": fraud_detection,
    "🗂️  Transaction History": transaction_history,
    "📈  Fraud Analytics": fraud_analytics,
    "🧭  Risk Analysis": risk_analysis,
    "🛠️  Admin Dashboard": admin_dashboard,
}

BRAND = ('<div class="brand"><div class="brand-logo">🛡️</div><div><div class="brand-name">UPI SENTINEL AI</div>'
         '<div class="brand-sub">Intelligent Digital Payment Fraud Detection &amp; Risk Intelligence</div></div></div>')


def login_screen(api: ApiClient) -> None:
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        ui._html('<div class="login-wrap"><div class="login-card">' + BRAND.replace("1.05rem", "1.2rem") +
                 '<p style="color:#9DB0D8;font-size:.88rem;margin:.2rem 0 .8rem 0">Sign in to the fraud-monitoring workspace.</p></div></div>')
        tab_in, tab_up = st.tabs(["Sign in", "Create account"])
        with tab_in:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("SIGN IN"):
                    try:
                        api.login(username.strip(), password)
                        st.rerun()
                    except ApiError as err:
                        ui.api_error(err)
            st.caption("Demo accounts are listed in the README / .env.example.")
        with tab_up:
            with st.form("register_form"):
                u = st.text_input("Choose a username", help="3-32 letters, digits, _ . -")
                e = st.text_input("Email (optional)")
                p1 = st.text_input("Password (min 8 chars, letters + digits)", type="password")
                p2 = st.text_input("Confirm password", type="password")
                if st.form_submit_button("CREATE ACCOUNT"):
                    if p1 != p2:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            api.register(u.strip(), p1, email=e.strip())
                            st.success("Account created. You can now sign in.")
                        except ApiError as err:
                            ui.api_error(err)
        try:
            api.health()
        except ApiError as err:
            st.warning(err.detail)


def main() -> None:
    ui.inject_css()
    api = ApiClient()
    if not api.token:
        login_screen(api)
        return

    user = api.user or {}
    if "risk_cfg" not in st.session_state:
        try:
            cfg = api.risk_config()
            st.session_state["risk_cfg"] = {"low_max": cfg["low"][1], "medium_max": cfg["medium"][1]}
        except ApiError:
            st.session_state["risk_cfg"] = {"low_max": 30, "medium_max": 70}

    with st.sidebar:
        ui._html(BRAND)
        ui._html(f'<div class="user-chip"><span class="status-dot"></span><b>{ui.esc(user.get("username", ""))}</b> '
                 f'&nbsp;·&nbsp; {ui.esc(user.get("role", "")).upper()}</div>')
        pages = [p for p in PAGES if user.get("role") == "admin" or "Admin" not in p]
        choice = st.radio("Navigation", pages, label_visibility="collapsed")
        st.write("")
        if st.button("Sign out"):
            api.logout()
            st.session_state.pop("risk_cfg", None)
            st.rerun()
        st.caption("Academic decision-support system. Trained on synthetic data; it never connects to banks "
                   "or blocks payments.")

    try:
        PAGES[choice].render(api)
    except ApiError as err:
        ui.api_error(err)
        if err.status == 401:
            st.rerun()


main()
