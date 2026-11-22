"""F. Admin dashboard - available to admin users only (also enforced by the API)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.api_client import ApiError
from frontend.components import ui_components as ui


def render(api) -> None:
    ui.page_header("Admin Dashboard", "Platform monitoring, users, model and database", "🛠️")
    if (api.user or {}).get("role") != "admin":
        st.error("Administrator access required.")
        return
    try:
        s = api.admin_stats()
    except ApiError as err:
        ui.api_error(err)
        return

    a, db = s["analytics"], s["database"]
    ui.kpi_row([
        dict(label="Total users", value=f"{s['users_total']}", icon="👥", tone="blue", hint=f"{s['users_by_role']['admin']} admin / {s['users_by_role']['user']} user"),
        dict(label="Transactions", value=f"{s['transactions_total']:,}", icon="🧾", tone="cyan"),
        dict(label="Suspicious", value=f"{s['suspicious_total']:,}", icon="⚠️", tone="red"),
        dict(label="Database", value=db["dialect"].upper(), icon="🗄️", tone="purple", hint=f"{db['size_kb']} KB" if db["size_kb"] is not None else "")])

    tabs = st.tabs(["Users", "Transaction monitoring", "Suspicious monitoring", "Fraud analytics", "Model performance", "Database"])
    with tabs[0]:
        df = pd.DataFrame(s["users"])[["id", "username", "full_name", "email", "role", "is_active", "created_at"]]
        ui.show_table(df)
        st.caption("Passwords are stored only as salted PBKDF2-SHA256 hashes and are never shown.")
    with tabs[1]:
        if not a["has_data"]:
            ui.empty_state("No transactions yet.")
        else:
            d = pd.DataFrame(a["daily"])
            ui.show_chart(ui.dual_axis(d["date"], d["total"], d["amount"], "Transactions", "Amount (INR)",
                                       title="Platform volume and value per day", line_color=ui.CYAN), "ad_daily")
            r = pd.DataFrame(a["recent"])
            r.columns = ["Transaction ID", "Date & time", "Amount", "Type", "Prediction", "Risk score", "Risk", "Anomaly"]
            ui.show_table(r)
    with tabs[2]:
        sus = pd.DataFrame(s["suspicious_recent"])
        if sus.empty:
            st.info("No suspicious transactions recorded.")
        else:
            sus = sus[["reference", "created_at", "amount", "transaction_type", "fraud_probability", "risk_score", "risk_level", "is_anomaly", "user_id"]]
            sus.columns = ["Transaction ID", "Date & time", "Amount", "Type", "Probability", "Risk score", "Risk", "Anomaly", "User ID"]
            ui.show_table(sus, column_config={"Probability": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                              "Amount": st.column_config.NumberColumn(format="₹%.2f")})
    with tabs[3]:
        if a["has_data"]:
            t = pd.DataFrame(a["type_breakdown"])
            h = pd.DataFrame(a["hourly"])
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Bar(x=t["transaction_type"], y=t["suspicious"], marker_color=ui.RED))
                ui.show_chart(ui.style_fig(fig, 320, "Suspicious by type", legend=False), "ad_type")
            with c2:
                fig = go.Figure(go.Bar(x=h["hour"], y=h["suspicious"], marker_color=ui.AMBER))
                ui.show_chart(ui.style_fig(fig, 320, "Suspicious by hour", legend=False), "ad_hour")
            k = a["kpis"]
            if k["labelled"]:
                st.caption(f"On {k['labelled']:,} demo transactions with known labels: recall {k['recall_on_labelled']:.3f}, "
                           f"precision {k['precision_on_labelled']:.3f}.")
    with tabs[4]:
        mp = pd.DataFrame(s["model_performance"])
        if mp.empty:
            st.info("Model performance has not been synced yet.")
        else:
            mp["model_name"] = mp.apply(lambda r: r["model_name"] + (" ★" if r["is_selected"] else ""), axis=1)
            ui.show_table(mp.drop(columns=["is_selected"]), column_config={
                c: st.column_config.NumberColumn(format="%.4f") for c in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold")})
    with tabs[5]:
        rows = pd.DataFrame({"Table": list(db["table_rows"]), "Rows": list(db["table_rows"].values())})
        c1, c2 = st.columns([1, 1])
        with c1:
            ui.show_table(rows)
        with c2:
            st.metric("Engine", db["dialect"])
            st.metric("File size", f"{db['size_kb']} KB" if db["size_kb"] is not None else "n/a")
            st.metric("Last prediction (UTC)", db["last_prediction_at"] or "none")
