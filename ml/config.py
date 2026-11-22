"""Central configuration shared by the ML, backend and frontend layers.

Values come from environment variables (see .env.example). Nothing secret lives here.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

try:  # python-dotenv is optional at import time; it is listed in requirements.txt
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:  # pragma: no cover
    pass

DATASET_DIR = ROOT_DIR / "dataset"
MODELS_DIR = ROOT_DIR / "models"
SYNTHETIC_DATASET_PATH = DATASET_DIR / "upi_synthetic.csv"
PAYSIM_DATASET_PATH = DATASET_DIR / "paysim.csv"
MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
RANDOM_STATE = 42


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


# Project-defined risk bands (NOT an official banking / NPCI standard).
#   LOW    : 0 .. RISK_LOW_MAX
#   MEDIUM : RISK_LOW_MAX+1 .. RISK_MEDIUM_MAX
#   HIGH   : RISK_MEDIUM_MAX+1 .. 100
RISK_LOW_MAX = _int_env("RISK_LOW_MAX", 30)
RISK_MEDIUM_MAX = _int_env("RISK_MEDIUM_MAX", 70)
if not (0 <= RISK_LOW_MAX < RISK_MEDIUM_MAX < 100):
    raise ValueError("Risk thresholds must satisfy 0 <= RISK_LOW_MAX < RISK_MEDIUM_MAX < 100")

# Expected share of anomalies for the Isolation Forest (unsupervised module).
ANOMALY_CONTAMINATION = _float_env("ANOMALY_CONTAMINATION", 0.05)
