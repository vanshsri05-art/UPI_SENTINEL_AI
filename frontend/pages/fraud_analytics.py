"""D. Fraud analytics - trends, types, hours, high-risk, amounts, model performance, anomalies."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.api_client import ApiError
from frontend.components import ui_components as ui


def render(api) -> None:
    ui.page_header("Fraud Analytics", "Trends, patterns and model quality", "📈")
    try:
        data = api.analytics()
    except ApiError as err:
        ui.api_error(err)
        return

    tabs = st.tabs(["Trends", "Types & hours", "High-risk", "Amounts", "Model performance", "Anomaly detection"])
    with tabs[0]:
        _trends(data)
    with tabs[1]:
        _types_hours(data)
    with tabs[2]:
        _high_risk(api, data)
    with tabs[3]:
        _amounts(api, data)
    with tabs[4]:
        _model_performance(api)
    with tabs[5]:
        _anomalies(api)


def _need_data(data) -> bool:
    if not data["has_data"]:
        ui.empty_state("No analysed transactions yet - run `python run.py seed` or use Fraud Detection.")
        return False
    return True


def _trends(data) -> None:
    if not _need_data(data):
        return
    d = pd.DataFrame(data["daily"])
    ui.show_chart(ui.dual_axis(d["date"], d["total"], d["suspicious"], "Transactions", "Suspicious",
                               title="Daily volume vs. suspicious count"), "fa_daily")
    fig = go.Figure(go.Scatter(x=d["date"], y=d["suspicious_rate"], mode="lines+markers", fill="tozeroy",
                               line=dict(color=ui.CYAN, width=3), fillcolor="rgba(0,194,255,.15)"))
    fig.update_yaxes(title_text="% flagged")
    ui.show_chart(ui.style_fig(fig, 320, "Share of transactions flagged per day", legend=False), "fa_rate")
    d["cum"] = d["suspicious"].cumsum()
    fig = go.Figure(go.Scatter(x=d["date"], y=d["cum"], mode="lines", line=dict(color=ui.RED, width=3, shape="spline")))
    ui.show_chart(ui.style_fig(fig, 300, "Cumulative suspicious transactions", legend=False), "fa_cum")


def _types_hours(data) -> None:
    if not _need_data(data):
        return
    c1, c2 = st.columns(2)
    t = pd.DataFrame(data["type_breakdown"])
    with c1:
        fig = go.Figure(go.Bar(x=t["transaction_type"], y=t["suspicious_rate"], marker_color=ui.AMBER,
                               text=[f"{v:.1f}%" for v in t["suspicious_rate"]], textposition="outside"))
        fig.update_yaxes(title_text="% of type flagged")
        ui.show_chart(ui.style_fig(fig, 340, "Suspicious rate by transaction type", legend=False), "fa_type_rate")
    with c2:
        fig = go.Figure(go.Pie(labels=t["transaction_type"], values=t["suspicious"], hole=0.55))
        ui.show_chart(ui.style_fig(fig, 340, "Where suspicious transactions come from"), "fa_type_pie")
    h = pd.DataFrame(data["hourly"])
    h["rate"] = (100 * h["suspicious"] / h["total"].where(h["total"] > 0)).fillna(0).round(2)
    fig = ui.dual_axis(h["hour"], h["total"], h["rate"], "Transactions", "% flagged",
                       title="Hourly transaction pattern and flag rate", bar_color=ui.BLUE, line_color=ui.RED)
    fig.update_xaxes(title_text="Hour of day", dtick=1)
    ui.show_chart(fig, "fa_hourly")


def _high_risk(api, data) -> None:
    if not _need_data(data):
        return
    rt = pd.DataFrame(data["risk_by_type"])
    fig = go.Figure()
    for level in ("Low", "Medium", "High"):
        fig.add_bar(x=rt["transaction_type"], y=rt[level], name=level, marker_color=ui.RISK_COLORS[level])
    fig.update_layout(barmode="stack")
    ui.show_chart(ui.style_fig(fig, 340, "Risk level mix by transaction type"), "fa_risk_type")
    try:
        tx = pd.DataFrame(api.transactions(limit=2000)["items"])
    except ApiError as err:
        ui.api_error(err)
        return
    fig = go.Figure()
    for pred, color in (("Genuine", ui.BLUE), ("Suspicious", ui.RED)):
        s = tx[tx["prediction"] == pred]
        fig.add_trace(go.Scattergl(x=s["amount"], y=s["risk_score"], mode="markers", name=pred,
                                   marker=dict(color=color, size=6, opacity=0.6), text=s["reference"]))
    fig.update_xaxes(type="log", title_text="Amount (INR, log scale)")
    fig.update_yaxes(title_text="Risk score")
    ui.show_chart(ui.style_fig(fig, 380, "Amount vs. risk score"), "fa_scatter")
    ui.section("Top high-risk transactions")
    hr = pd.DataFrame(data["high_risk_top"])
    if hr.empty:
        st.info("No transactions in the High band.")
    else:
        hr.columns = ["Transaction ID", "Date & time", "Amount", "Type", "Prediction", "Risk score", "Risk", "Anomaly"]
        ui.show_table(hr, column_config={"Amount": st.column_config.NumberColumn(format="₹%.2f"),
                                         "Risk score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d")})


def _amounts(api, data) -> None:
    if not _need_data(data):
        return
    k = data["kpis"]
    ui.kpi_row([dict(label="Total amount", value=ui.inr(k["total_amount"]), icon="💰", tone="blue"),
                dict(label="Flagged amount", value=ui.inr(k["suspicious_amount"]), icon="🛑", tone="red",
                     hint=f"{100 * k['suspicious_amount'] / max(k['total_amount'], 1):.1f}% of value"),
                dict(label="Avg. amount", value=ui.inr(k["total_amount"] / max(k["total"], 1)), icon="📊", tone="cyan")])
    a = pd.DataFrame(data["amount_distribution"])
    fig = go.Figure()
    fig.add_bar(x=a["range"], y=a["count"], name="All", marker_color=ui.BLUE)
    fig.add_bar(x=a["range"], y=a["suspicious"], name="Suspicious", marker_color=ui.RED)
    ui.show_chart(ui.style_fig(fig, 340, "Transactions per amount band (INR)"), "fa_amount_band")
    try:
        tx = pd.DataFrame(api.transactions(limit=2000)["items"])
    except ApiError as err:
        ui.api_error(err)
        return
    fig = go.Figure()
    for pred, color in (("Genuine", ui.BLUE), ("Suspicious", ui.RED)):
        fig.add_trace(go.Box(y=tx.loc[tx["prediction"] == pred, "amount"], name=pred, marker_color=color, boxmean=True))
    fig.update_yaxes(type="log", title_text="Amount (INR, log)")
    ui.show_chart(ui.style_fig(fig, 340, "Amount spread: genuine vs. suspicious", legend=False), "fa_amount_box")


def _model_performance(api) -> None:
    try:
        m = api.model_performance()
    except ApiError as err:
        ui.api_error(err)
        return
    st.caption(f"{m['dataset']['note']}  Trained {m['trained_at']}. Split: {m['split']}. "
               f"Selection: {m['selection_criterion']}")
    models = m["models"]
    sel = next(x for x in models if x["is_selected"])
    ui.kpi_row([dict(label="Selected model", value=sel["name"], icon="🏆", tone="blue"),
                dict(label="Precision", value=f"{sel['test']['precision']:.3f}", icon="🎯", tone="cyan", hint="Test set"),
                dict(label="Recall", value=f"{sel['test']['recall']:.3f}", icon="🔎", tone="green", hint="Test set"),
                dict(label="ROC-AUC", value=f"{sel['test']['roc_auc']:.3f}", icon="📐", tone="purple", hint="Test set")])
    rows = [{"Model": x["name"] + (" ★" if x["is_selected"] else ""), **{k.upper() if k == "f1" else k.replace("_", "-").title(): x["test"][k]
            for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")}} for x in models]
    df = pd.DataFrame(rows)
    ui.show_table(df, column_config={c: st.column_config.NumberColumn(format="%.4f") for c in df.columns if c != "Model"})

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig = go.Figure()
    for x in models:
        fig.add_bar(name=x["name"], x=[k.upper() if k == "f1" else k.replace("_", "-").title() for k in metrics],
                    y=[x["test"][k] for k in metrics])
    fig.update_layout(barmode="group")
    ui.show_chart(ui.style_fig(fig, 340, "Model comparison (test set)"), "fa_model_cmp")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for x in models:
            r = x["curves"]["roc"]
            fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines", name=f"{x['name']} (AUC {x['test']['roc_auc']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#6C80AD"), name="Chance"))
        fig.update_xaxes(title_text="False positive rate")
        fig.update_yaxes(title_text="True positive rate")
        ui.show_chart(ui.style_fig(fig, 360, "ROC curves"), "fa_roc")
    with c2:
        fig = go.Figure()
        for x in models:
            p = x["curves"]["pr"]
            fig.add_trace(go.Scatter(x=p["recall"], y=p["precision"], mode="lines", name=x["name"]))
        fig.update_xaxes(title_text="Recall")
        fig.update_yaxes(title_text="Precision")
        ui.show_chart(ui.style_fig(fig, 360, "Precision-recall curves"), "fa_pr")

    choice = st.selectbox("Confusion matrix for", [x["name"] for x in models],
                          index=[x["name"] for x in models].index(sel["name"]))
    chosen = next(x for x in models if x["name"] == choice)
    cm = chosen["test"]["confusion_matrix"]
    fig = go.Figure(go.Heatmap(z=cm, x=["Pred. Genuine", "Pred. Suspicious"], y=["Actual Genuine", "Actual Fraud"],
                               text=cm, texttemplate="%{text}", textfont=dict(size=20),
                               colorscale=[[0, "#0F1A3A"], [1, ui.BLUE]], showscale=False))
    fig.update_yaxes(autorange="reversed")
    ui.show_chart(ui.style_fig(fig, 320, f"Confusion matrix - {choice} (threshold {chosen['threshold']:.3f})", legend=False), "fa_cm")
    rep = pd.DataFrame(chosen["test"]["classification_report"]).T.round(4)
    rep.index.name = "class"
    ui.section("Classification report")
    ui.show_table(rep.reset_index())


def _anomalies(api) -> None:
    try:
        a = api.anomalies()
    except ApiError as err:
        ui.api_error(err)
        return
    ui.callout("Isolation Forest is unsupervised: it flags transactions that are statistically unusual. "
               "An anomaly is NOT automatically fraud, so it is shown separately from the classifier.", "amber")
    ui.kpi_row([dict(label="Anomalies stored", value=f"{a['total_anomalies']:,}", icon="🔍", tone="amber"),
                dict(label="Also flagged suspicious", value=f"{a['also_flagged_suspicious']:,}", icon="🤝", tone="red",
                     hint=f"in the {a['shown']} shown"),
                dict(label="Anomaly only", value=f"{a['anomaly_only']:,}", icon="❓", tone="purple",
                     hint="Unusual, not flagged as fraud")])
    c = a["offline_comparison"]
    ui.section("Isolation Forest vs. supervised classifier (held-out test set)")
    cmp_df = pd.DataFrame({
        "Method": ["Isolation Forest (unsupervised)", "Classifier (supervised)"],
        "Flagged": [c["anomalies_flagged"], c["supervised_flagged"]],
        "Precision": [c["anomaly_precision"], c["supervised_precision"]],
        "Recall": [c["anomaly_recall"], c["supervised_recall"]]})
    ui.show_table(cmp_df, column_config={"Precision": st.column_config.NumberColumn(format="%.3f"),
                                         "Recall": st.column_config.NumberColumn(format="%.3f")})
    fig = go.Figure(go.Bar(x=["Both", "Anomaly only", "Classifier only", "Neither"],
                           y=[c["flagged_by_both"], c["anomaly_only"], c["supervised_only"], c["flagged_by_neither"]],
                           marker_color=[ui.RED, ui.AMBER, ui.BLUE, "#41537F"]))
    fig.update_yaxes(type="log", title_text="Transactions (log scale)")
    ui.show_chart(ui.style_fig(fig, 320, "Agreement between the two methods", legend=False), "fa_anom_overlap")
    st.caption(f"Real fraud among 'anomaly-only' transactions: {c['actual_fraud_in_anomaly_only']} of {c['anomaly_only']}; "
               f"among 'classifier-only': {c['actual_fraud_in_supervised_only']} of {c['supervised_only']}.")
    if a["items"]:
        ui.section("Most recent anomalies")
        df = pd.DataFrame(a["items"])[["reference", "created_at", "amount", "transaction_type", "anomaly_score", "prediction", "risk_score"]]
        df.columns = ["Transaction ID", "Date & time", "Amount", "Type", "Anomaly score", "Classifier says", "Risk score"]
        ui.show_table(df, column_config={"Amount": st.column_config.NumberColumn(format="₹%.2f")})
