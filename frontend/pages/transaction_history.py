"""C. Transaction history with search, date / risk / type filters and CSV export."""
from datetime import date

import pandas as pd
import streamlit as st

from frontend.api_client import ApiError
from frontend.components import ui_components as ui

TYPES = ["P2P", "P2M", "BILL_PAYMENT", "QR_PAYMENT", "COLLECT_REQUEST", "CASH_OUT", "CASH_IN"]


def render(api) -> None:
    ui.page_header("Transaction History", "Every analysed transaction, searchable and exportable", "🗂️")

    with st.container():
        f1, f2, f3 = st.columns([1.3, 1.2, 1])
        search = f1.text_input("Search (ID or type)", placeholder="e.g. TXN2611 or P2P")
        dates = f2.date_input("Date range", value=(), format="YYYY-MM-DD")
        prediction = f3.selectbox("Prediction", ["All", "Genuine", "Suspicious"])
        g1, g2, g3 = st.columns([1.3, 1.2, 1])
        risk = g1.multiselect("Risk level", ["Low", "Medium", "High"])
        types = g2.multiselect("Transaction type", TYPES)
        limit = g3.selectbox("Max rows", [500, 1000, 2000, 5000], index=1)

    date_from = date_to = None
    if isinstance(dates, (tuple, list)):
        if len(dates) >= 1:
            date_from = dates[0]
        if len(dates) == 2:
            date_to = dates[1]
    elif dates:
        date_from = dates

    try:
        data = api.transactions(search=search.strip() or None, risk_level=risk, transaction_type=types,
                                prediction=None if prediction == "All" else prediction,
                                date_from=date_from.isoformat() if date_from else None,
                                date_to=(date_to or date_from).isoformat() if date_from else None, limit=limit)
    except ApiError as err:
        ui.api_error(err)
        return

    items = data["items"]
    if not items:
        ui.empty_state("No transactions match these filters.")
        return
    df = pd.DataFrame(items)
    df["created_at"] = pd.to_datetime(df["created_at"])
    st.caption(f"{len(df):,} of {data['total']:,} matching transactions shown "
               f"({'all users' if data['scope'] == 'all' else 'your account'}).")

    export = df[["reference", "created_at", "amount", "transaction_type", "prediction", "fraud_probability",
                 "risk_score", "risk_level", "is_anomaly"]].rename(columns={
        "reference": "transaction_id", "created_at": "date_time"})
    st.download_button("⬇️  Export CSV", export.to_csv(index=False).encode("utf-8"),
                       file_name=f"upi_sentinel_transactions_{date.today().isoformat()}.csv", mime="text/csv")

    view = df[["reference", "created_at", "amount", "transaction_type", "prediction", "risk_score",
               "risk_level", "is_anomaly"]].copy()
    view["prediction"] = view["prediction"].map({"Suspicious": "🔴 Suspicious", "Genuine": "🟢 Genuine"})
    view["risk_level"] = view["risk_level"].map({"Low": "🟢 Low", "Medium": "🟠 Medium", "High": "🔴 High"})
    view.columns = ["Transaction ID", "Date & time", "Amount", "Type", "Prediction", "Risk score", "Risk category", "Anomaly"]
    ui.show_table(view, height=560, column_config={
        "Amount": st.column_config.NumberColumn(format="₹%.2f"),
        "Risk score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
        "Date & time": st.column_config.DatetimeColumn(format="DD MMM YYYY, HH:mm"),
        "Anomaly": st.column_config.CheckboxColumn(disabled=True)})
