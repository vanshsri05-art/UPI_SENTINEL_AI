"""Unsupervised anomaly detection with Isolation Forest.

The model never sees the fraud label. An anomaly means "statistically unusual compared with the
training population" - it is NOT proof of fraud. Results are therefore reported separately from the
supervised fraud prediction and compared with it.
"""
from __future__ import annotations

from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ml.config import ANOMALY_CONTAMINATION, ANOMALY_MODEL_PATH, RANDOM_STATE


def fit_anomaly_model(X_train: pd.DataFrame, contamination: float = ANOMALY_CONTAMINATION) -> Dict:
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train)
    raw = -model.score_samples(X_train)  # higher = more unusual
    return {
        "model": model,
        "quantiles": np.quantile(raw, np.linspace(0, 1, 101)),
        "contamination": float(contamination),
        "feature_columns": list(X_train.columns),
    }


def score_anomalies(bundle: Dict, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return (is_anomaly flags, anomaly score 0-100 = percentile vs. the training population)."""
    X = X[bundle["feature_columns"]]
    raw = -bundle["model"].score_samples(X)
    scores = np.interp(raw, bundle["quantiles"], np.linspace(0, 100, 101))
    flags = bundle["model"].predict(X) == -1
    return flags, scores


def save_anomaly_model(bundle: Dict, path=ANOMALY_MODEL_PATH) -> None:
    joblib.dump(bundle, path, compress=3)


def load_anomaly_model(path=ANOMALY_MODEL_PATH) -> Dict:
    return joblib.load(path)


def compare_with_supervised(y_true, supervised_flags, anomaly_flags) -> Dict:
    """Cross-tabulate Isolation Forest anomalies against supervised predictions and true labels."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(supervised_flags).astype(bool)
    a = np.asarray(anomaly_flags).astype(bool)

    def prec(flag):
        return float(y[flag].mean()) if flag.sum() else 0.0

    def rec(flag):
        return float(flag[y].mean()) if y.sum() else 0.0

    both, a_only, s_only = a & s, a & ~s, s & ~a
    return {
        "n_transactions": int(len(y)),
        "actual_fraud": int(y.sum()),
        "anomalies_flagged": int(a.sum()),
        "supervised_flagged": int(s.sum()),
        "flagged_by_both": int(both.sum()),
        "anomaly_only": int(a_only.sum()),
        "supervised_only": int(s_only.sum()),
        "flagged_by_neither": int((~a & ~s).sum()),
        "actual_fraud_in_anomaly_only": int(y[a_only].sum()),
        "actual_fraud_in_supervised_only": int(y[s_only].sum()),
        "anomaly_precision": prec(a),
        "anomaly_recall": rec(a),
        "supervised_precision": prec(s),
        "supervised_recall": rec(s),
        "recall_of_either": rec(a | s),
        "note": ("Anomalies are unusual transactions, not confirmed fraud. The supervised model "
                 "learns from labelled fraud; Isolation Forest only measures how atypical a "
                 "transaction is, so the two lists overlap but are not the same."),
    }
