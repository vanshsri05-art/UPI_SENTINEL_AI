"""B. Fraud detection - sends the form to POST /predict (real trained model, no fake scores)."""
import streamlit as st

from frontend.api_client import ApiError
from frontend.components import ui_components as ui

TYPES = ["P2P", "P2M", "BILL_PAYMENT", "QR_PAYMENT", "COLLECT_REQUEST", "CASH_OUT", "CASH_IN"]
TYPE_HELP = {"P2P": "Person to person", "P2M": "Person to merchant", "BILL_PAYMENT": "Utility / bill payment",
             "QR_PAYMENT": "QR scan payment", "COLLECT_REQUEST": "Collect (pull) request",
             "CASH_OUT": "Cash withdrawal", "CASH_IN": "Cash deposit"}

DEFAULTS = {"fd_amount": 2500.0, "fd_type": "P2P", "fd_hour": 14, "fd_freq": 2, "fd_sender": 50000.0,
            "fd_receiver": 20000.0, "fd_device": 0.10, "fd_new_device": False, "fd_new_payee": False,
            "fd_failed": 0}

# Input presets only fill the form. The verdict always comes from the trained model.
PRESETS = {
    "Everyday grocery payment (QR)": dict(fd_amount=640.0, fd_type="QR_PAYMENT", fd_hour=18, fd_freq=2,
        fd_sender=24000.0, fd_receiver=85000.0, fd_device=0.05, fd_new_device=False, fd_new_payee=False, fd_failed=0),
    "Large transfer to a known contact": dict(fd_amount=15000.0, fd_type="P2P", fd_hour=14, fd_freq=3,
        fd_sender=120000.0, fd_receiver=60000.0, fd_device=0.10, fd_new_device=False, fd_new_payee=False, fd_failed=0),
    "Late-night collect request, new payee": dict(fd_amount=38000.0, fd_type="COLLECT_REQUEST", fd_hour=2, fd_freq=9,
        fd_sender=41000.0, fd_receiver=0.0, fd_device=0.72, fd_new_device=True, fd_new_payee=True, fd_failed=3),
}


def _apply_preset() -> None:
    preset = PRESETS.get(st.session_state.get("fd_preset"))
    if preset:
        st.session_state.update(preset)


def render(api) -> None:
    ui.page_header("Fraud Detection", "Analyse a payment with the trained machine-learning model", "🛡️")
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)

    st.selectbox("Load a sample scenario (optional)", ["Custom (keep my values)"] + list(PRESETS),
                 key="fd_preset", on_change=_apply_preset)

    with st.form("analyze_form"):
        ui.section("Transaction details")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("Transaction amount (₹)", min_value=1.0, max_value=1_000_000.0, step=100.0, key="fd_amount")
            st.selectbox("Transaction type", TYPES, key="fd_type", format_func=lambda t: f"{t} — {TYPE_HELP[t]}")
        with c2:
            st.slider("Transaction hour (0–23)", 0, 23, key="fd_hour")
            st.number_input("Transaction frequency (sender txns, last 24h)", min_value=0, max_value=500, step=1, key="fd_freq")
        with c3:
            st.number_input("Sender balance before payment (₹)", min_value=0.0, max_value=1e9, step=1000.0, key="fd_sender")
            st.number_input("Receiver balance (₹)", min_value=0.0, max_value=1e9, step=1000.0, key="fd_receiver")
        ui.section("Device & session risk indicators")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.slider("Device / session risk score", 0.0, 1.0, step=0.01, key="fd_device")
        with d2:
            st.toggle("New / unrecognised device", key="fd_new_device")
        with d3:
            st.toggle("New payee (first payment)", key="fd_new_payee")
        with d4:
            st.number_input("Failed attempts (24h)", min_value=0, max_value=50, step=1, key="fd_failed")
        submitted = st.form_submit_button("🔍  ANALYZE TRANSACTION")

    if submitted:
        payload = {
            "amount": float(st.session_state.fd_amount), "transaction_type": st.session_state.fd_type,
            "transaction_hour": int(st.session_state.fd_hour), "transaction_frequency": int(st.session_state.fd_freq),
            "sender_balance": float(st.session_state.fd_sender), "receiver_balance": float(st.session_state.fd_receiver),
            "device_risk_score": float(st.session_state.fd_device), "is_new_device": bool(st.session_state.fd_new_device),
            "is_new_payee": bool(st.session_state.fd_new_payee), "failed_attempts_24h": int(st.session_state.fd_failed)}
        try:
            with st.spinner("Scoring with the model..."):
                st.session_state["fd_result"] = api.predict(payload)
        except ApiError as err:
            st.session_state.pop("fd_result", None)
            ui.api_error(err)

    result = st.session_state.get("fd_result")
    if result:
        _show_result(result)


def _show_result(r: dict) -> None:
    suspicious = r["prediction"] == "Suspicious"
    color = ui.RED if suspicious else ui.GREEN
    cfg = st.session_state.get("risk_cfg", {"low_max": 30, "medium_max": 70})
    ui.section("Analysis result")
    left, right = st.columns([1, 1.15])
    with left:
        ui.show_chart(ui.risk_gauge(r["risk_score"], cfg["low_max"], cfg["medium_max"]), "fd_gauge")
    with right:
        icon = "⚠️" if suspicious else "✅"
        ui._html(f'<div class="verdict" style="border-color:{color}88;background:{color}14">'
                 f'<p class="v-title" style="color:{color}">{icon} {r["prediction"].upper()}</p>'
                 f'<div class="v-sub">Fraud probability <b>{r["fraud_probability"] * 100:.1f}%</b> &nbsp;|&nbsp; '
                 f'Risk score <b>{r["risk_score"]}/100</b> &nbsp;|&nbsp; {ui.risk_badge(r["risk_level"])}</div></div>')
        st.write("")
        m1, m2 = st.columns(2)
        m1.metric("Transaction reference", r["transaction_id"])
        m2.metric("Model", r["model_name"])
        st.caption(f"Analysed at {r['timestamp']} (UTC). Explanation method: {r['explanation_method']}.")
        a = r["anomaly"]
        ui.callout(("Anomaly check (separate, unsupervised): this transaction is statistically UNUSUAL "
                    if a["is_anomaly"] else "Anomaly check (separate, unsupervised): this transaction looks typical ")
                   + f"(anomaly score {a['anomaly_score']:.0f}/100). An anomaly is not proof of fraud.",
                   "amber" if a["is_anomaly"] else "green")
    ui.callout(r["summary"], "red" if suspicious else "green")
    if r["assumed_defaults"]:
        st.caption("Fields not supplied, typical values assumed: " + ", ".join(r["assumed_defaults"]))

    ui.section("What contributed to this result")
    c1, c2 = st.columns([1, 1])
    with c1:
        ui.show_chart(ui.factor_chart(r["top_factors"]), "fd_factors")
    with c2:
        ui.factor_list(r["top_factors"])
    st.caption("Decision-support output from a model trained on synthetic data. It does not block or "
               "approve payments.")
