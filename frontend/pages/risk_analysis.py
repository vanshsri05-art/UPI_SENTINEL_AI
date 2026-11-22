"""E. Risk analysis - project-defined bands, what-if thresholds, global and local explanations."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.api_client import ApiError
from frontend.components import ui_components as ui


def _level(score: float, low: int, med: int) -> str:
    return "Low" if score <= low else ("Medium" if score <= med else "High")


def render(api) -> None:
    ui.page_header("Risk Analysis", "How risk scores map to Low / Medium / High, and why", "🧭")
    try:
        data = api.analytics()
        perf = api.model_performance()
    except ApiError as err:
        ui.api_error(err)
        return
    th = data["risk_thresholds"]
    low, med = th["low_max"], th["medium_max"]

    ui.callout("Risk bands are PROJECT-DEFINED thresholds for this academic system - they are not an official "
               "banking or NPCI standard. Configure them with RISK_LOW_MAX / RISK_MEDIUM_MAX in .env.", "amber")
    dist = data["risk_distribution"]
    total = max(sum(dist.values()), 1)
    ui.kpi_row([
        dict(label=f"LOW RISK (0–{low})", value=f"{dist['Low']:,}", icon="🟢", tone="green", hint=f"{100 * dist['Low'] / total:.1f}% of transactions"),
        dict(label=f"MEDIUM RISK ({low + 1}–{med})", value=f"{dist['Medium']:,}", icon="🟠", tone="amber", hint=f"{100 * dist['Medium'] / total:.1f}% of transactions"),
        dict(label=f"HIGH RISK ({med + 1}–100)", value=f"{dist['High']:,}", icon="🔴", tone="red", hint=f"{100 * dist['High'] / total:.1f}% of transactions")])

    if data["has_data"]:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Pie(labels=list(dist), values=list(dist.values()), hole=0.6,
                                   marker=dict(colors=[ui.RISK_COLORS[k] for k in dist], line=dict(color="#0A1128", width=3))))
            ui.show_chart(ui.style_fig(fig, 330, "Risk category distribution", legend=False), "ra_pie")
        with c2:
            s = pd.DataFrame(data["score_histogram"])
            fig = go.Figure()
            fig.add_bar(x=s["range"], y=s["count"], name="All", marker_color=ui.BLUE)
            fig.add_bar(x=s["range"], y=s["suspicious"], name="Flagged suspicious", marker_color=ui.RED)
            fig.update_layout(barmode="overlay")
            fig.update_xaxes(title_text="Risk score")
            ui.show_chart(ui.style_fig(fig, 330, "Risk score histogram"), "ra_hist")

        ui.section("What-if: try different thresholds")
        st.caption("Exploration only - it re-bins the stored scores in your browser session. To change the real "
                   "bands, edit .env and restart the API.")
        w1, w2 = st.columns(2)
        new_low = w1.slider("Low risk up to", 5, 60, low)
        new_med = w2.slider("Medium risk up to", new_low + 5 if new_low + 5 <= 95 else 95, 95, max(med, min(new_low + 5, 95)))
        try:
            tx = pd.DataFrame(api.transactions(limit=5000)["items"])
            lv = tx["risk_score"].apply(lambda v: _level(v, new_low, new_med)).value_counts().reindex(
                ["Low", "Medium", "High"], fill_value=0)
            fig = go.Figure(go.Bar(x=lv.index, y=lv.values, marker_color=[ui.RISK_COLORS[k] for k in lv.index],
                                   text=lv.values, textposition="outside"))
            ui.show_chart(ui.style_fig(fig, 300, f"Distribution with Low ≤ {new_low}, Medium ≤ {new_med}", legend=False), "ra_whatif")
        except ApiError as err:
            ui.api_error(err)

        rt = pd.DataFrame(data["risk_by_type"])
        fig = go.Figure()
        for level in ("Low", "Medium", "High"):
            fig.add_bar(x=rt["transaction_type"], y=rt[level], name=level, marker_color=ui.RISK_COLORS[level])
        fig.update_layout(barmode="stack")
        ui.show_chart(ui.style_fig(fig, 320, "Risk level by transaction type"), "ra_type")
    else:
        ui.empty_state("No analysed transactions yet.")

    ui.section("Global model explanation")
    gi = perf["global_importance"]
    items = gi["items"][:12][::-1]
    fig = go.Figure(go.Bar(x=[i["importance"] for i in items], y=[i["label"] for i in items], orientation="h",
                           marker_color=ui.CYAN))
    fig.update_xaxes(title_text="Importance")
    ui.show_chart(ui.style_fig(fig, 420, f"Most important features - {gi['method']}", legend=False), "ra_global")
    st.caption(f"Local explanations use: {perf['explainer_method']}. SHAP is used automatically for tree models "
               f"when the `shap` package is installed; otherwise a model-agnostic fallback is used.")

    if data["has_data"]:
        ui.section("Explain a specific transaction")
        try:
            tx = pd.DataFrame(api.transactions(limit=300)["items"]).sort_values("risk_score", ascending=False)
        except ApiError as err:
            ui.api_error(err)
            return
        ref = st.selectbox("Transaction (highest risk first)", tx["reference"].tolist(),
                           format_func=lambda r: f"{r}  —  score {int(tx.loc[tx['reference'] == r, 'risk_score'].iloc[0])}")
        try:
            d = api.transaction_detail(ref)
        except ApiError as err:
            ui.api_error(err)
            return
        color = ui.RED if d["prediction"] == "Suspicious" else ui.GREEN
        ui._html(f'<div class="verdict" style="border-color:{color}88;background:{color}14"><p class="v-title" style="color:{color}">'
                 f'{d["prediction"].upper()}</p><div class="v-sub">Risk score <b>{d["risk_score"]}</b> | probability '
                 f'<b>{d["fraud_probability"] * 100:.1f}%</b> | {ui.risk_badge(d["risk_level"])} | anomaly: '
                 f'{"yes" if d["is_anomaly"] else "no"}</div></div>')
        exp = d.get("explanation")
        if exp:
            ui.callout(exp["summary"], "red" if d["prediction"] == "Suspicious" else "green")
            c1, c2 = st.columns(2)
            with c1:
                ui.show_chart(ui.factor_chart(exp["factors"]), "ra_local")
            with c2:
                ui.factor_list(exp["factors"])
