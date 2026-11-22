"""Inference: uses exactly the same `build_features` as training, then adds risk, anomaly and
explanation information.

    from ml.predict import get_predictor
    result = get_predictor().predict_one({"amount": 25000, "transaction_type": "P2P", ...})
"""
from __future__ import annotations

import json
import threading
import warnings
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn

from ml import config
from ml.anomaly_detection import load_anomaly_model, score_anomalies
from ml.explainability import TransactionExplainer, summarize
from ml.preprocessing import (
    FEATURE_COLUMNS, OPTIONAL_DEFAULTS, RAW_FEATURE_COLUMNS, build_features)


class ModelNotFoundError(RuntimeError):
    """Raised when the trained artefacts are missing or unreadable."""


def risk_level_from_score(score: float, low_max: int = config.RISK_LOW_MAX,
                          medium_max: int = config.RISK_MEDIUM_MAX) -> str:
    """LOW: 0..low_max, MEDIUM: low_max+1..medium_max, HIGH: above. (Project-defined bands.)"""
    if score <= low_max:
        return "Low"
    if score <= medium_max:
        return "Medium"
    return "High"


class FraudPredictor:
    def __init__(self, model_path=config.MODEL_PATH, metadata_path=config.METADATA_PATH,
                 anomaly_path=config.ANOMALY_MODEL_PATH):
        for p in (model_path, metadata_path, anomaly_path):
            if not p.exists():
                raise ModelNotFoundError(
                    f"Missing {p.name}. Train the model first:  python run.py train")
        try:
            self.model = joblib.load(model_path)
            self.anomaly_bundle = load_anomaly_model(anomaly_path)
        except Exception as exc:  # version mismatch etc.
            raise ModelNotFoundError(
                f"Could not load the saved model ({exc}). Re-train on this machine: python run.py train"
            ) from exc
        self.metadata: Dict = json.loads(metadata_path.read_text())
        if self.metadata.get("feature_columns") != FEATURE_COLUMNS:
            raise ModelNotFoundError(
                "Saved model was trained with a different feature set. Re-train: python run.py train")
        trained_with = self.metadata.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            warnings.warn(f"Model was trained with scikit-learn {trained_with}, running {sklearn.__version__}. "
                          "Re-train if you see errors.")
        self.model_name: str = self.metadata["selected_model"]
        self.threshold: float = float(self.metadata["decision_threshold"])
        self._explainer: Optional[TransactionExplainer] = None
        self._lock = threading.Lock()

    # ---- helpers ---------------------------------------------------------------------
    @property
    def explainer(self) -> TransactionExplainer:
        with self._lock:
            if self._explainer is None:
                self._explainer = TransactionExplainer(
                    self.model, FEATURE_COLUMNS, self.metadata["baseline"])
            return self._explainer

    @staticmethod
    def _raw_frame(records: List[Dict]) -> pd.DataFrame:
        rows = []
        for rec in records:
            row = {**OPTIONAL_DEFAULTS, **{k: v for k, v in rec.items() if v is not None}}
            missing = [c for c in RAW_FEATURE_COLUMNS if c not in row]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")
            rows.append({c: row[c] for c in RAW_FEATURE_COLUMNS})
        return pd.DataFrame(rows, columns=RAW_FEATURE_COLUMNS)

    # ---- API -------------------------------------------------------------------------
    def predict_records(self, records: List[Dict], explain: bool = False) -> List[Dict]:
        raw = self._raw_frame(records)
        X = build_features(raw)
        proba = self.model.predict_proba(X)[:, 1]
        a_flags, a_scores = score_anomalies(self.anomaly_bundle, X)

        out = []
        for i in range(len(raw)):
            p = float(proba[i])
            score = int(round(p * 100))
            suspicious = p >= self.threshold
            item = {
                "prediction": "Suspicious" if suspicious else "Genuine",
                "fraud_probability": round(p, 4),
                "risk_score": score,
                "risk_level": risk_level_from_score(score),
                "is_anomaly": bool(a_flags[i]),
                "anomaly_score": round(float(a_scores[i]), 1),
                "model_name": self.model_name,
            }
            if explain:
                exp = self.explainer.explain(X.iloc[[i]], raw.iloc[i].to_dict())
                item["explanation_method"] = exp["method"]
                item["top_factors"] = exp["factors"]
                item["summary"] = summarize(exp["factors"], suspicious)
            out.append(item)
        return out

    def predict_one(self, record: Dict, explain: bool = True) -> Dict:
        return self.predict_records([record], explain=explain)[0]


_predictor: Optional[FraudPredictor] = None
_predictor_lock = threading.Lock()


def get_predictor(reload: bool = False) -> FraudPredictor:
    global _predictor
    with _predictor_lock:
        if _predictor is None or reload:
            _predictor = FraudPredictor()
        return _predictor
