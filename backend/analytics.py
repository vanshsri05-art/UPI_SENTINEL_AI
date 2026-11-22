"""Pure-pandas analytics. Everything is computed from the stored prediction data passed in -
nothing is hardcoded. Input: DataFrame from services.query_transactions."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

AMOUNT_BINS = [0, 500, 1000, 5000, 10000, 25000, 50000, 100000, float("inf")]
AMOUNT_LABELS = ["<500", "500-1K", "1K-5K", "5K-10K", "10K-25K", "25K-50K", "50K-1L", ">=1L"]
RISK_LEVELS = ["Low", "Medium", "High"]


def _rate(part, whole) -> float:
    return round(100.0 * float(part) / float(whole), 2) if whole else 0.0


def empty_analytics() -> Dict[str, Any]:
    return {
        "has_data": False,
        "kpis": {"total": 0, "genuine": 0, "suspicious": 0, "fraud_detection_rate": 0.0,
                 "total_amount": 0.0, "suspicious_amount": 0.0, "high_risk": 0, "anomalies": 0,
                 "avg_risk_score": 0.0, "labelled": 0, "recall_on_labelled": None,
                 "precision_on_labelled": None},
        "daily": [], "hourly": [{"hour": h, "total": 0, "suspicious": 0} for h in range(24)],
        "type_breakdown": [], "amount_distribution": [], "risk_distribution": {k: 0 for k in RISK_LEVELS},
        "score_histogram": [], "risk_by_type": [], "recent": [], "high_risk_top": [],
    }


def compute_analytics(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return empty_analytics()

    d = df.copy()
    d["created_at"] = pd.to_datetime(d["created_at"])
    d["is_susp"] = (d["prediction"] == "Suspicious").astype(int)
    total = len(d)
    suspicious = int(d["is_susp"].sum())

    labelled = d[d["actual_label"].notna()]
    recall = precision = None
    if len(labelled):
        actual = labelled["actual_label"].astype(int)
        flagged = labelled["is_susp"].astype(int)
        tp = int(((actual == 1) & (flagged == 1)).sum())
        recall = round(tp / max(int((actual == 1).sum()), 1), 4)
        precision = round(tp / max(int((flagged == 1).sum()), 1), 4)

    kpis = {
        "total": int(total),
        "genuine": int(total - suspicious),
        "suspicious": suspicious,
        "fraud_detection_rate": _rate(suspicious, total),  # share of analysed txns flagged
        "total_amount": round(float(d["amount"].sum()), 2),
        "suspicious_amount": round(float(d.loc[d["is_susp"] == 1, "amount"].sum()), 2),
        "high_risk": int((d["risk_level"] == "High").sum()),
        "anomalies": int(d["is_anomaly"].sum()),
        "avg_risk_score": round(float(d["risk_score"].mean()), 1),
        "labelled": int(len(labelled)),
        "recall_on_labelled": recall,
        "precision_on_labelled": precision,
    }

    d["date"] = d["created_at"].dt.strftime("%Y-%m-%d")
    g = d.groupby("date").agg(total=("reference", "count"), suspicious=("is_susp", "sum"),
                              amount=("amount", "sum")).reset_index().sort_values("date")
    daily = [{"date": r.date, "total": int(r.total), "suspicious": int(r.suspicious),
              "amount": round(float(r.amount), 2), "suspicious_rate": _rate(r.suspicious, r.total)}
             for r in g.itertuples()]

    h = d.groupby("transaction_hour").agg(total=("reference", "count"), suspicious=("is_susp", "sum"))
    h = h.reindex(range(24), fill_value=0)
    hourly = [{"hour": int(i), "total": int(r.total), "suspicious": int(r.suspicious)}
              for i, r in h.iterrows()]

    t = d.groupby("transaction_type").agg(total=("reference", "count"), suspicious=("is_susp", "sum"),
                                          amount=("amount", "sum")).reset_index()
    type_breakdown = [{"transaction_type": r.transaction_type, "total": int(r.total),
                       "suspicious": int(r.suspicious), "amount": round(float(r.amount), 2),
                       "suspicious_rate": _rate(r.suspicious, r.total)}
                      for r in t.sort_values("suspicious", ascending=False).itertuples()]

    d["amount_bin"] = pd.cut(d["amount"], bins=AMOUNT_BINS, labels=AMOUNT_LABELS, right=False)
    a = d.groupby("amount_bin", observed=False).agg(count=("reference", "count"),
                                                   suspicious=("is_susp", "sum")).reset_index()
    amount_distribution = [{"range": str(r.amount_bin), "count": int(r.count),
                            "suspicious": int(r.suspicious)} for r in a.itertuples()]

    risk_distribution = {k: int((d["risk_level"] == k).sum()) for k in RISK_LEVELS}

    edges = list(range(0, 101, 10))
    d["score_bin"] = pd.cut(d["risk_score"], bins=edges + [101], right=False, labels=[
        f"{lo}-{lo + 9}" if lo < 100 else "100" for lo in edges])
    s = d.groupby("score_bin", observed=False).agg(count=("reference", "count"),
                                                  suspicious=("is_susp", "sum")).reset_index()
    score_histogram = [{"range": str(r.score_bin), "count": int(r.count),
                        "suspicious": int(r.suspicious)} for r in s.itertuples()]

    rbt = pd.crosstab(d["transaction_type"], d["risk_level"]).reindex(columns=RISK_LEVELS, fill_value=0)
    risk_by_type = [{"transaction_type": idx, **{k: int(row[k]) for k in RISK_LEVELS}}
                    for idx, row in rbt.iterrows()]

    cols = ["reference", "created_at", "amount", "transaction_type", "prediction", "risk_score",
            "risk_level", "is_anomaly"]

    def rows(frame: pd.DataFrame):
        out = []
        for r in frame[cols].itertuples(index=False):
            out.append({"reference": r.reference, "created_at": r.created_at.isoformat(timespec="seconds"),
                        "amount": float(r.amount), "transaction_type": r.transaction_type,
                        "prediction": r.prediction, "risk_score": int(r.risk_score),
                        "risk_level": r.risk_level, "is_anomaly": bool(r.is_anomaly)})
        return out

    recent = rows(d.sort_values("created_at", ascending=False).head(10))
    high_risk_top = rows(d[d["risk_level"] == "High"].sort_values(
        ["risk_score", "amount"], ascending=False).head(10))

    return {"has_data": True, "kpis": kpis, "daily": daily, "hourly": hourly,
            "type_breakdown": type_breakdown, "amount_distribution": amount_distribution,
            "risk_distribution": risk_distribution, "score_histogram": score_histogram,
            "risk_by_type": risk_by_type, "recent": recent, "high_risk_top": high_risk_top}
