"""A. Overview dashboard - every number comes from /analytics (computed from stored data)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.api_client import ApiError
from frontend.components import ui_components as ui


def render(api) -> None:
    ui.page_header("Overview Dashboard", "Live summary of every payment analysed by the model", "📊")
    try:
        data = api.analytics()
    except ApiError as err:
        ui.api_error(err)
        return
    if not data["has_data"]:
        ui.empty_state("Analyse a transaction on the Fraud Detection page, or load demo data with "
                       "`python run.py seed`.")
        return

    k = data["kpis"]
    scope = "all users" if data.get("scope") == "all" else "your account"
    st.caption(f"Showing analytics for {scope}. 'Fraud detection rate' = share of analysed transactions "
               f"the model flagged as suspicious.")
    ui.kpi_row([
        dict(label="Transactions analysed", value=f"{k['total']:,}", icon="🧾", tone="blue", hint="All stored predictions"),
        dict(label="Genuine", value=f"{k['genuine']:,}", icon="✅", tone="green", hint=f"{100 - k['fraud_detection_rate']:.1f}% of total"),
        dict(label="Suspicious", value=f"{k['suspicious']:,}", icon="⚠️", tone="red", hint="Model decision threshold"),
        dict(label="Fraud detection rate", value=f"{k['fraud_detection_rate']:.2f}%", icon="🎯", tone="cyan", hint="Flagged / analysed"),
    ])
    ui.kpi_row([
        dict(label="Total amount", value=ui.inr(k["total_amount"]), icon="💰", tone="blue", hint="Sum of analysed payments"),
        dict(label="Suspicious amount", value=ui.inr(k["suspicious_amount"]), icon="🛑", tone="amber", hint="Value of flagged payments"),
        dict(label="High-risk (score > %d)" % data["risk_thresholds"]["medium_max"], value=f"{k['high_risk']:,}",
             icon="🚨", tone="red", hint="Project-defined band"),
        dict(label="Avg. risk score", value=f"{k['avg_risk_score']:.1f}", icon="📈", tone="purple", hint="0 – 100"),
    ])

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Pie(labels=["Genuine", "Suspicious"], values=[k["genuine"], k["suspicious"]], hole=0.62,
                               marker=dict(colors=[ui.GREEN, ui.RED], line=dict(color="#0A1128", width=3)),
                               textinfo="percent+label"))
        fig.add_annotation(text=f"<b>{k['total']:,}</b><br>analysed", showarrow=False, font=dict(size=16))
        ui.show_chart(ui.style_fig(fig, 340, "Genuine vs. suspicious", legend=False), "ov_pie")
    with c2:
        d = pd.DataFrame(data["daily"])
        fig = ui.dual_axis(d["date"], d["total"], d["suspicious"], "Transactions", "Suspicious", title="Daily transaction trend")
        ui.show_chart(fig, "ov_daily")

    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure(go.Scatter(x=d["date"], y=d["suspicious_rate"], mode="lines+markers", fill="tozeroy",
                                   line=dict(color=ui.CYAN, width=3), fillcolor="rgba(0,194,255,.15)", name="Flag rate %"))
        fig.update_yaxes(title_text="% flagged")
        ui.show_chart(ui.style_fig(fig, 340, "Fraud detection trend (daily % flagged)", legend=False), "ov_trend")
    with c4:
        a = pd.DataFrame(data["amount_distribution"])
        fig = go.Figure()
        fig.add_bar(x=a["range"], y=a["count"], name="All", marker_color=ui.BLUE)
        fig.add_bar(x=a["range"], y=a["suspicious"], name="Suspicious", marker_color=ui.RED)
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title_text="Amount (INR)")
        ui.show_chart(ui.style_fig(fig, 340, "Transaction amount distribution"), "ov_amount")

    c5, c6 = st.columns(2)
    with c5:
        t = pd.DataFrame(data["type_breakdown"])
        fig = go.Figure(go.Bar(x=t["transaction_type"], y=t["suspicious"], marker_color=ui.RED,
                               text=[f"{r:.1f}%" for r in t["suspicious_rate"]], textposition="outside"))
        fig.update_yaxes(title_text="Suspicious count")
        ui.show_chart(ui.style_fig(fig, 340, "Fraud by transaction type (label = % of type flagged)", legend=False), "ov_type")
    with c6:
        h = pd.DataFrame(data["hourly"])
        h["rate"] = (100 * h["suspicious"] / h["total"].where(h["total"] > 0)).fillna(0).round(2)
        fig = ui.dual_axis(h["hour"], h["suspicious"], h["rate"], "Suspicious", "% flagged", title="Hourly fraud distribution",
                           bar_color=ui.RED, line_color=ui.CYAN)
        fig.update_xaxes(title_text="Hour of day", dtick=2)
        ui.show_chart(fig, "ov_hour")

    ui.section("Recent transaction activity")
    r = pd.DataFrame(data["recent"])
    r["created_at"] = pd.to_datetime(r["created_at"])
    r["prediction"] = r["prediction"].map({"Suspicious": "🔴 Suspicious", "Genuine": "🟢 Genuine"})
    r["risk_level"] = r["risk_level"].map({"Low": "🟢 Low", "Medium": "🟠 Medium", "High": "🔴 High"})
    r = r.rename(columns={"reference": "Transaction ID", "created_at": "Date & time", "amount": "Amount",
                          "transaction_type": "Type", "prediction": "Prediction", "risk_score": "Risk score",
                          "risk_level": "Risk", "is_anomaly": "Anomaly"})
    ui.show_table(r, column_config={
        "Amount": st.column_config.NumberColumn(format="₹%.2f"),
        "Risk score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
        "Date & time": st.column_config.DatetimeColumn(format="DD MMM YYYY, HH:mm")})
