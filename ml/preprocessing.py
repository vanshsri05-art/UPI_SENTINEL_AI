"""Data loading, cleaning and feature engineering.

The SAME `build_features` function is used for training and for live prediction, which
guarantees identical preprocessing and identical column ordering (FEATURE_COLUMNS).

Leakage policy
--------------
Only information that would be known *before / at the moment* a payment is analysed is used:
amount, type, hour, sender activity so far, balances BEFORE the transaction and device/session
signals. Post-transaction balances (e.g. PaySim's newbalanceOrig / newbalanceDest), the label,
and any future information are never used as inputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import PAYSIM_DATASET_PATH, RANDOM_STATE, SYNTHETIC_DATASET_PATH

TARGET_COLUMN = "is_fraud"

TRANSACTION_TYPES = [
    "P2P",
    "P2M",
    "BILL_PAYMENT",
    "QR_PAYMENT",
    "COLLECT_REQUEST",
    "CASH_OUT",
    "CASH_IN",
]

RAW_FEATURE_COLUMNS = [
    "amount",
    "transaction_type",
    "transaction_hour",
    "transaction_frequency",
    "sender_balance",
    "receiver_balance",
    "device_risk_score",
    "is_new_device",
    "is_new_payee",
    "failed_attempts_24h",
]
NUMERIC_RAW_COLUMNS = [c for c in RAW_FEATURE_COLUMNS if c != "transaction_type"]

NIGHT_HOURS = (23, 0, 1, 2, 3, 4, 5)

# Final model input columns, in the exact order used by training and prediction.
FEATURE_COLUMNS = (
    [
        "amount",
        "amount_log",
        "transaction_hour",
        "is_night",
        "transaction_frequency",
        "sender_balance",
        "receiver_balance",
        "amount_to_balance_ratio",
        "receiver_balance_zero",
        "device_risk_score",
        "is_new_device",
        "is_new_payee",
        "failed_attempts_24h",
    ]
    + [f"type_{t}" for t in TRANSACTION_TYPES]
)

# Human-friendly names for UI / explanations.
FEATURE_LABELS: Dict[str, str] = {
    "amount": "Transaction amount",
    "amount_log": "Transaction amount (log scale)",
    "transaction_hour": "Hour of day",
    "is_night": "Night-time transaction",
    "transaction_frequency": "Sender activity (txns in last 24h)",
    "sender_balance": "Sender balance",
    "receiver_balance": "Receiver balance",
    "amount_to_balance_ratio": "Amount vs. sender balance",
    "receiver_balance_zero": "Empty / new receiver account",
    "device_risk_score": "Device / session risk score",
    "is_new_device": "New device",
    "is_new_payee": "New payee",
    "failed_attempts_24h": "Failed attempts (24h)",
}
for _t in TRANSACTION_TYPES:
    FEATURE_LABELS[f"type_{_t}"] = f"Type = {_t}"


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
PAYSIM_TYPE_MAP = {
    "TRANSFER": "P2P",
    "PAYMENT": "P2M",
    "CASH_OUT": "CASH_OUT",
    "CASH_IN": "CASH_IN",
    "DEBIT": "BILL_PAYMENT",
}


def load_paysim(path: Path = PAYSIM_DATASET_PATH, max_rows: Optional[int] = 200_000,
                random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Map the public PaySim CSV onto this project's schema.

    PaySim is SYNTHETIC *mobile-money* data - it is NOT UPI data. Mapping notes:
      * hour              = step % 24 (one step = one hour)
      * sender_balance    = oldbalanceOrg   (balance BEFORE the transaction)
      * receiver_balance  = oldbalanceDest  (balance BEFORE the transaction)
      * frequency         = number of PRIOR transactions by the same sender in the file (past only)
      * device/session signals do not exist in PaySim -> set to 0 (constant, carries no signal)
      * newbalanceOrig / newbalanceDest are deliberately dropped (post-transaction => leakage)
    To keep training fast, all fraud rows are kept and non-fraud rows are down-sampled so the file
    has at most `max_rows` rows.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"PaySim file not found at {path}. Download it from Kaggle (search 'PaySim synthetic "
            f"financial dataset'), rename it to 'paysim.csv' and put it in the dataset/ folder."
        )
    raw = pd.read_csv(path)
    needed = {"step", "type", "amount", "nameOrig", "oldbalanceOrg", "oldbalanceDest", "isFraud"}
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError(f"PaySim file is missing columns: {sorted(missing)}")

    raw = raw.sort_values("step", kind="stable").reset_index(drop=True)
    raw["transaction_frequency"] = raw.groupby("nameOrig").cumcount()  # prior txns only
    mapped = pd.DataFrame(
        {
            "amount": raw["amount"],
            "transaction_type": raw["type"].map(PAYSIM_TYPE_MAP),
            "transaction_hour": raw["step"] % 24,
            "transaction_frequency": raw["transaction_frequency"],
            "sender_balance": raw["oldbalanceOrg"],
            "receiver_balance": raw["oldbalanceDest"],
            "device_risk_score": 0.0,
            "is_new_device": 0,
            "is_new_payee": 0,
            "failed_attempts_24h": 0,
            TARGET_COLUMN: raw["isFraud"],
        }
    )
    if max_rows and len(mapped) > max_rows:
        fraud = mapped[mapped[TARGET_COLUMN] == 1]
        legit = mapped[mapped[TARGET_COLUMN] == 0]
        n_legit = max(max_rows - len(fraud), 1)
        legit = legit.sample(n=min(n_legit, len(legit)), random_state=random_state)
        mapped = pd.concat([fraud, legit]).sample(frac=1.0, random_state=random_state)
    return mapped.reset_index(drop=True)


def load_dataset(source: str = "synthetic", path: Optional[Path] = None,
                 max_rows: Optional[int] = 200_000) -> pd.DataFrame:
    """Load the raw dataset. `source` is 'synthetic' (default) or 'paysim'."""
    if source == "paysim":
        return load_paysim(path or PAYSIM_DATASET_PATH, max_rows=max_rows)
    if source == "synthetic":
        p = Path(path or SYNTHETIC_DATASET_PATH)
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Create it with:  python dataset/generate_sample_data.py"
            )
        return pd.read_csv(p)
    raise ValueError("source must be 'synthetic' or 'paysim'")


# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------
def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Remove duplicates, handle missing values, validate ranges.

    Policy: rows missing a *critical* field (amount, type, label) are dropped; other missing
    numeric fields are filled with 0 (meaning "no / unknown"). Fixed constants are used instead
    of column statistics so that no information leaks from the test split.
    """
    missing_cols = [c for c in RAW_FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    report: Dict[str, int] = {"rows_in": int(len(df))}
    d = df.copy()
    report["missing_values_in"] = int(d[RAW_FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum().sum())

    before = len(d)
    d = d.drop_duplicates()
    report["duplicates_removed"] = int(before - len(d))

    d["transaction_type"] = d["transaction_type"].astype(str).str.strip().str.upper()
    before = len(d)
    d = d[d["transaction_type"].isin(TRANSACTION_TYPES)]
    report["unknown_type_removed"] = int(before - len(d))

    for col in NUMERIC_RAW_COLUMNS + [TARGET_COLUMN]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    before = len(d)
    d = d.dropna(subset=["amount", TARGET_COLUMN])
    d = d[d["amount"] > 0].copy()
    report["invalid_critical_removed"] = int(before - len(d))

    d[NUMERIC_RAW_COLUMNS[1:]] = d[NUMERIC_RAW_COLUMNS[1:]].fillna(0.0)
    d["transaction_hour"] = d["transaction_hour"].round().clip(0, 23).astype(int)
    d["device_risk_score"] = d["device_risk_score"].clip(0.0, 1.0)
    for flag in ("is_new_device", "is_new_payee"):
        d[flag] = (d[flag] > 0).astype(int)
    for col in ("transaction_frequency", "failed_attempts_24h", "sender_balance", "receiver_balance"):
        d[col] = d[col].clip(lower=0)
    d[TARGET_COLUMN] = d[TARGET_COLUMN].astype(int)

    d = d.reset_index(drop=True)
    report["rows_out"] = int(len(d))
    report["fraud_rows"] = int(d[TARGET_COLUMN].sum())
    return d, report


# --------------------------------------------------------------------------------------
# Feature engineering (shared by training and prediction)
# --------------------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw transaction columns into model features, ordered exactly as FEATURE_COLUMNS."""
    missing = [c for c in RAW_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing raw feature columns: {missing}")

    d = df[RAW_FEATURE_COLUMNS].copy()
    for col in NUMERIC_RAW_COLUMNS:
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0).astype(float)
    ttype = d["transaction_type"].astype(str).str.strip().str.upper()

    out = pd.DataFrame(index=d.index)
    out["amount"] = d["amount"].clip(lower=0.0)
    out["amount_log"] = np.log1p(out["amount"])
    out["transaction_hour"] = d["transaction_hour"].round().clip(0, 23)
    out["is_night"] = out["transaction_hour"].isin(NIGHT_HOURS).astype(int)
    out["transaction_frequency"] = d["transaction_frequency"].clip(lower=0.0)
    out["sender_balance"] = d["sender_balance"].clip(lower=0.0)
    out["receiver_balance"] = d["receiver_balance"].clip(lower=0.0)
    out["amount_to_balance_ratio"] = (out["amount"] / (out["sender_balance"] + 1.0)).clip(upper=100.0)
    out["receiver_balance_zero"] = (out["receiver_balance"] <= 0).astype(int)
    out["device_risk_score"] = d["device_risk_score"].clip(0.0, 1.0)
    out["is_new_device"] = (d["is_new_device"] > 0).astype(int)
    out["is_new_payee"] = (d["is_new_payee"] > 0).astype(int)
    out["failed_attempts_24h"] = d["failed_attempts_24h"].clip(lower=0.0)
    for t in TRANSACTION_TYPES:
        out[f"type_{t}"] = (ttype == t).astype(int)

    return out[FEATURE_COLUMNS].astype(float)


def split_data(df: pd.DataFrame, random_state: int = RANDOM_STATE
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70 / 15 / 15 train / validation / test split (raw cleaned rows)."""
    train_val, test = train_test_split(
        df, test_size=0.15, stratify=df[TARGET_COLUMN], random_state=random_state
    )
    train, val = train_test_split(
        train_val, test_size=0.15 / 0.85, stratify=train_val[TARGET_COLUMN], random_state=random_state
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# Values used by the API when optional fields are omitted. They describe a "typical, low-risk"
# user and are reported back to the caller as `assumed_defaults` so nothing is hidden.
OPTIONAL_DEFAULTS = {
    "sender_balance": 50000.0,
    "receiver_balance": 20000.0,
    "device_risk_score": 0.1,
    "is_new_device": 0,
    "is_new_payee": 0,
    "failed_attempts_24h": 0,
}
