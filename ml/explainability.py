"""Explainable AI: SHAP (tree models) with a model-agnostic fallback.

Global importance : mean |SHAP| over a sample; fallback = permutation importance.
Local explanation : SHAP values for one transaction; fallback = group-occlusion (replace a feature
                    group with its training-set median and measure the drop in fraud log-odds).
Both paths return the same structure, and the method used is always reported.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from ml.config import RANDOM_STATE
from ml.preprocessing import FEATURE_LABELS, TRANSACTION_TYPES

try:  # SHAP is optional: the app works (with a fallback) if it is not installed.
    import shap  # type: ignore

    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover
    shap = None
    SHAP_AVAILABLE = False

_TREE_MODEL_NAMES = {"DecisionTreeClassifier", "RandomForestClassifier", "XGBClassifier",
                     "ExtraTreesClassifier", "GradientBoostingClassifier"}

# Features that describe the same real-world fact are explained together.
FEATURE_GROUPS: Dict[str, List[str]] = {
    "amount": ["amount", "amount_log"],
    "amount_to_balance_ratio": ["amount_to_balance_ratio"],
    "time_of_day": ["transaction_hour", "is_night"],
    "transaction_type": [f"type_{t}" for t in TRANSACTION_TYPES],
    "transaction_frequency": ["transaction_frequency"],
    "sender_balance": ["sender_balance"],
    "receiver_balance": ["receiver_balance"],
    "receiver_balance_zero": ["receiver_balance_zero"],
    "device_risk_score": ["device_risk_score"],
    "is_new_device": ["is_new_device"],
    "is_new_payee": ["is_new_payee"],
    "failed_attempts_24h": ["failed_attempts_24h"],
}
GROUP_LABELS = {
    "amount": "Transaction amount",
    "amount_to_balance_ratio": "Amount vs. sender balance",
    "time_of_day": "Time of day",
    "transaction_type": "Transaction type",
    "transaction_frequency": "Sender activity (24h)",
    "sender_balance": "Sender balance",
    "receiver_balance": "Receiver balance",
    "receiver_balance_zero": "Empty / new receiver",
    "device_risk_score": "Device / session risk",
    "is_new_device": "New device",
    "is_new_payee": "New payee",
    "failed_attempts_24h": "Failed attempts (24h)",
}


def _inr(v: float) -> str:
    return f"\u20b9{float(v):,.0f}"


# Plain-language sentences: (raw transaction dict, increases_risk) -> text
_TEMPLATES = {
    "amount": lambda r, up: f"The amount ({_inr(r['amount'])}) is "
        + ("unusually large for this kind of payment." if up else "modest for this kind of payment."),
    "amount_to_balance_ratio": lambda r, up: (
        f"The payment is {r['amount'] / (r['sender_balance'] + 1):.0%} of the sender's balance "
        f"({_inr(r['sender_balance'])}), " + ("which resembles an account being drained." if up
                                              else "a normal share of the balance.")),
    "time_of_day": lambda r, up: f"It happens at {int(r['transaction_hour']):02d}:00, "
        + ("an hour when fraud is more common in the training data." if up
           else "an hour with typical, low-risk activity."),
    "transaction_type": lambda r, up: f"{r['transaction_type']} transactions "
        + ("carry a higher fraud rate in the training data." if up
           else "have a comparatively low fraud rate in the training data."),
    "transaction_frequency": lambda r, up: (
        f"The sender made {int(r['transaction_frequency'])} transactions in the last 24h, "
        + ("a burst of activity." if up else "a normal pace.")),
    "sender_balance": lambda r, up: f"The sender balance ({_inr(r['sender_balance'])}) "
        + ("adds to the risk profile." if up else "is consistent with legitimate use."),
    "receiver_balance": lambda r, up: f"The receiver balance ({_inr(r['receiver_balance'])}) "
        + ("adds to the risk profile." if up else "is consistent with an established account."),
    "receiver_balance_zero": lambda r, up: "The receiver account "
        + ("is empty / newly active, a common trait of mule accounts." if up
           else "already holds funds."),
    "device_risk_score": lambda r, up: f"The device/session risk score is {r['device_risk_score']:.2f}, "
        + ("which is elevated." if up else "which is low."),
    "is_new_device": lambda r, up: "The payment came from "
        + ("a device not seen before." if up else "a known device."),
    "is_new_payee": lambda r, up: "The receiver is "
        + ("a new payee for this sender." if up else "an existing payee."),
    "failed_attempts_24h": lambda r, up: f"There were {int(r['failed_attempts_24h'])} failed attempt(s) "
        + ("in the last 24h, which is suspicious." if up else "in the last 24h."),
}


def _display_value(key: str, r: Dict) -> str:
    return {
        "amount": _inr(r["amount"]),
        "amount_to_balance_ratio": f"{r['amount'] / (r['sender_balance'] + 1):.0%} of balance",
        "time_of_day": f"{int(r['transaction_hour']):02d}:00",
        "transaction_type": str(r["transaction_type"]),
        "transaction_frequency": str(int(r["transaction_frequency"])),
        "sender_balance": _inr(r["sender_balance"]),
        "receiver_balance": _inr(r["receiver_balance"]),
        "receiver_balance_zero": "Yes" if r["receiver_balance"] <= 0 else "No",
        "device_risk_score": f"{r['device_risk_score']:.2f}",
        "is_new_device": "Yes" if r["is_new_device"] else "No",
        "is_new_payee": "Yes" if r["is_new_payee"] else "No",
        "failed_attempts_24h": str(int(r["failed_attempts_24h"])),
    }[key]


def _is_tree_model(model) -> bool:
    return (not isinstance(model, Pipeline)) and type(model).__name__ in _TREE_MODEL_NAMES


def _positive_class_values(sv) -> np.ndarray:
    """Normalise the different shapes SHAP returns across versions to (n_rows, n_features)."""
    if isinstance(sv, list):  # old API: [class0, class1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:  # new API: (rows, features, classes)
        sv = sv[:, :, 1]
    return sv


# --------------------------------------------------------------------------------------
# Global importance
# --------------------------------------------------------------------------------------
def compute_global_importance(model, X: pd.DataFrame, y: Optional[pd.Series] = None,
                              max_rows: int = 1000) -> Dict:
    """Return {'method': ..., 'items': [{feature, label, importance}] sorted high -> low}."""
    if len(X) > max_rows:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X), max_rows, replace=False)
        Xs = X.iloc[idx]
        ys = None if y is None else y.iloc[idx]
    else:
        Xs, ys = X, y

    method, values = None, None
    if SHAP_AVAILABLE and _is_tree_model(model):
        try:
            sv = _positive_class_values(shap.TreeExplainer(model).shap_values(Xs))
            values = np.abs(sv).mean(axis=0)
            method = "SHAP (mean |SHAP value|)"
        except Exception:
            values = None
    if values is None:
        if ys is None or ys.sum() == 0:
            raise ValueError("Permutation importance needs labels containing fraud cases.")
        result = permutation_importance(model, Xs, ys, scoring="average_precision",
                                        n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
        values = np.clip(result.importances_mean, 0, None)
        method = "Permutation importance (drop in PR-AUC)"

    items = [{"feature": c, "label": FEATURE_LABELS.get(c, c), "importance": float(v)}
             for c, v in zip(Xs.columns, values)]
    items.sort(key=lambda d: d["importance"], reverse=True)
    return {"method": method, "items": items}


# --------------------------------------------------------------------------------------
# Local explanation
# --------------------------------------------------------------------------------------
class TransactionExplainer:
    """Explains single predictions. Create once per loaded model and reuse."""

    def __init__(self, model, feature_columns: Sequence[str], baseline: Dict[str, float]):
        self.model = model
        self.columns = list(feature_columns)
        self.baseline = pd.Series(baseline).reindex(self.columns).fillna(0.0)
        self._tree_explainer = None
        if SHAP_AVAILABLE and _is_tree_model(model):
            try:
                self._tree_explainer = shap.TreeExplainer(model)
            except Exception:
                self._tree_explainer = None
        self.method = "SHAP (TreeExplainer)" if self._tree_explainer is not None else "Occlusion (fallback)"

    # ---- raw per-group contributions -------------------------------------------------
    def _group_contributions(self, X_row: pd.DataFrame) -> Tuple[Dict[str, float], str]:
        if self._tree_explainer is not None:
            try:
                sv = _positive_class_values(self._tree_explainer.shap_values(X_row))[0]
                per_feature = dict(zip(self.columns, sv))
                return ({g: float(sum(per_feature[c] for c in cols)) for g, cols in FEATURE_GROUPS.items()},
                        "SHAP (TreeExplainer)")
            except Exception:
                pass  # fall through to the occlusion method
        groups = list(FEATURE_GROUPS.items())
        masked = pd.concat([X_row] * len(groups), ignore_index=True)
        for i, (_, cols) in enumerate(groups):
            masked.loc[i, cols] = self.baseline[cols].values
        s0 = float(self._log_odds(X_row)[0])
        s_masked = self._log_odds(masked)
        return {g: float(s0 - s_masked[i]) for i, (g, _) in enumerate(groups)}, "Occlusion (fallback)"

    def _log_odds(self, X: pd.DataFrame) -> np.ndarray:
        """Model output on the log-odds scale (stays informative when probabilities saturate)."""
        if hasattr(self.model, "decision_function"):
            return np.asarray(self.model.decision_function(X), dtype=float)
        p = np.clip(self.model.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    def explain(self, X_row: pd.DataFrame, raw: Dict, top_k: int = 6) -> Dict:
        contributions, method = self._group_contributions(X_row)
        total = sum(abs(v) for v in contributions.values()) or 1.0
        factors = []
        for key, value in contributions.items():
            up = value > 0
            factors.append({
                "feature": key,
                "label": GROUP_LABELS[key],
                "value": _display_value(key, raw),
                "contribution": round(value, 5),
                "relative_impact": round(100.0 * value / total, 1),
                "direction": "increases risk" if up else "reduces risk",
                "text": _TEMPLATES[key](raw, up),
            })
        factors.sort(key=lambda f: abs(f["contribution"]), reverse=True)
        return {"method": method, "factors": factors[:top_k]}


def summarize(factors: List[Dict], is_suspicious: bool) -> str:
    """One plain-language sentence built from the strongest factors."""
    ups = [f for f in factors if f["contribution"] > 0][:3]
    downs = [f for f in factors if f["contribution"] < 0][:3]
    if is_suspicious:
        if ups:
            return "Flagged as suspicious mainly because: " + " ".join(f["text"] for f in ups)
        return "Flagged as suspicious, but no single factor dominates the model's decision."
    if downs:
        return "Looks genuine mainly because: " + " ".join(f["text"] for f in downs)
    return "Looks genuine; no single factor dominates the model's decision."
