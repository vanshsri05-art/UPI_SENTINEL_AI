"""ML / security / analytics unit tests (no web framework required)."""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset"))

from generate_sample_data import generate  # noqa: E402
from ml import config  # noqa: E402
from ml.preprocessing import (FEATURE_COLUMNS, RAW_FEATURE_COLUMNS, TARGET_COLUMN,  # noqa: E402
                              build_features, clean_dataset, split_data)
from ml.predict import risk_level_from_score  # noqa: E402

SAFE = {"amount": 450, "transaction_type": "P2M", "transaction_hour": 13, "transaction_frequency": 2,
        "sender_balance": 30000, "receiver_balance": 50000, "device_risk_score": 0.05}
RISKY = {"amount": 48000, "transaction_type": "COLLECT_REQUEST", "transaction_hour": 2,
         "transaction_frequency": 9, "sender_balance": 50000, "receiver_balance": 0,
         "device_risk_score": 0.7, "is_new_device": 1, "is_new_payee": 1, "failed_attempts_24h": 3}


@pytest.fixture(scope="module")
def small_df():
    return generate(n_rows=4000, fraud_rate=0.05, seed=7)


def test_feature_order_and_no_target_leakage(small_df):
    X = build_features(small_df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert not X.isna().any().any()
    banned = ("is_fraud", "isfraud", "newbalance", "transaction_id", "label")
    assert not any(b in c.lower() for c in FEATURE_COLUMNS for b in banned)


def test_build_features_rejects_missing_columns(small_df):
    with pytest.raises(ValueError):
        build_features(small_df.drop(columns=["amount"]))


def test_clean_dataset_removes_duplicates_and_invalid_rows(small_df):
    dirty = pd.concat([small_df, small_df.iloc[:10]], ignore_index=True)
    dirty.loc[100, "amount"] = np.nan        # rows 100+ are not part of the duplicated block
    dirty.loc[101, "amount"] = -5
    dirty.loc[102, "transaction_type"] = "UNKNOWN"
    dirty.loc[103, "sender_balance"] = np.nan
    clean, report = clean_dataset(dirty)
    assert report["duplicates_removed"] == 10
    assert report["invalid_critical_removed"] == 2 and report["unknown_type_removed"] == 1
    assert report["rows_out"] == len(clean) == len(small_df) - 3
    assert (clean["amount"] > 0).all() and not clean[RAW_FEATURE_COLUMNS].isna().any().any()


def test_split_is_stratified_and_disjoint(small_df):
    clean, _ = clean_dataset(small_df)
    tr, va, te = split_data(clean)
    assert len(tr) + len(va) + len(te) == len(clean)
    rates = [d[TARGET_COLUMN].mean() for d in (tr, va, te)]
    assert max(rates) - min(rates) < 0.01
    ids = [set(d["transaction_id"]) for d in (tr, va, te)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_models_learn_signal(small_df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    clean, _ = clean_dataset(small_df)
    tr, _, te = split_data(clean)
    model = RandomForestClassifier(n_estimators=60, class_weight="balanced_subsample", random_state=0, n_jobs=-1)
    model.fit(build_features(tr), tr[TARGET_COLUMN])
    auc = roc_auc_score(te[TARGET_COLUMN], model.predict_proba(build_features(te))[:, 1])
    assert auc > 0.8


def test_isolation_forest_scores_in_range(small_df):
    from ml.anomaly_detection import fit_anomaly_model, score_anomalies
    X = build_features(clean_dataset(small_df)[0])
    bundle = fit_anomaly_model(X)
    flags, scores = score_anomalies(bundle, X)
    assert scores.min() >= 0 and scores.max() <= 100
    assert 0.02 < flags.mean() < 0.10


def test_risk_level_boundaries():
    assert [risk_level_from_score(s, 30, 70) for s in (0, 30, 31, 70, 71, 100)] == \
           ["Low", "Low", "Medium", "Medium", "High", "High"]
    assert config.RISK_LOW_MAX < config.RISK_MEDIUM_MAX


needs_model = pytest.mark.skipif(not config.MODEL_PATH.exists(),
                                 reason="train first: python run.py train")


@needs_model
def test_predictor_outputs_are_consistent():
    from ml.predict import get_predictor
    p = get_predictor()
    assert p.metadata["feature_columns"] == FEATURE_COLUMNS
    safe, risky = p.predict_one(SAFE), p.predict_one(RISKY)
    for r in (safe, risky):
        assert 0.0 <= r["fraud_probability"] <= 1.0
        assert r["risk_score"] == round(r["fraud_probability"] * 100)
        assert r["risk_level"] == risk_level_from_score(r["risk_score"])
        assert r["prediction"] in ("Genuine", "Suspicious")
        assert len(r["top_factors"]) > 0 and r["summary"]
    assert risky["fraud_probability"] > safe["fraud_probability"]
    assert risky["prediction"] == "Suspicious" and safe["prediction"] == "Genuine"
    assert p.predict_one(SAFE)["fraud_probability"] == safe["fraud_probability"]  # deterministic


@needs_model
def test_predict_batch_matches_single():
    from ml.predict import get_predictor
    p = get_predictor()
    batch = p.predict_records([SAFE, RISKY])
    assert batch[0]["fraud_probability"] == p.predict_one(SAFE, explain=False)["fraud_probability"]


@needs_model
def test_missing_required_field_raises():
    from ml.predict import get_predictor
    with pytest.raises(ValueError):
        get_predictor().predict_one({"transaction_type": "P2P"})


def test_password_hashing_and_tokens():
    os.environ["SECRET_KEY"] = "t" * 48
    from backend.security import (create_access_token, decode_access_token, hash_password,
                                  validate_password_strength, verify_password, LoginThrottle)
    h = hash_password("Sample@123")
    assert "Sample@123" not in h and h.startswith("pbkdf2_sha256$")
    assert verify_password("Sample@123", h) and not verify_password("wrong", h)
    assert hash_password("Sample@123") != h  # unique salt
    assert decode_access_token(create_access_token("bob", "user"))["role"] == "user"
    with pytest.raises(ValueError):
        validate_password_strength("short1")
    with pytest.raises(ValueError):
        validate_password_strength("onlyletters")
    t = LoginThrottle(max_attempts=2, window_seconds=60)
    t.record_failure("x"); t.record_failure("X")
    assert t.is_blocked("x")
    t.reset("x")
    assert not t.is_blocked("x")


def test_secret_key_is_required(monkeypatch):
    from backend.security import get_secret_key
    monkeypatch.setenv("SECRET_KEY", "short")
    with pytest.raises(RuntimeError):
        get_secret_key()


def test_analytics_from_data():
    from backend.analytics import compute_analytics
    now = datetime(2026, 11, 20, 12)
    df = pd.DataFrame({
        "reference": ["A", "B", "C", "D"], "created_at": [now, now, now - timedelta(days=1), now - timedelta(days=1)],
        "amount": [100.0, 60000.0, 2500.0, 900.0], "transaction_type": ["P2P", "P2P", "P2M", "QR_PAYMENT"],
        "transaction_hour": [10, 2, 14, 18], "prediction": ["Genuine", "Suspicious", "Genuine", "Genuine"],
        "risk_score": [5, 95, 20, 40], "risk_level": ["Low", "High", "Low", "Medium"],
        "is_anomaly": [False, True, False, False], "actual_label": [0, 1, 0, 1]})
    a = compute_analytics(df)
    k = a["kpis"]
    assert (k["total"], k["genuine"], k["suspicious"], k["high_risk"], k["anomalies"]) == (4, 3, 1, 1, 1)
    assert k["fraud_detection_rate"] == 25.0 and k["recall_on_labelled"] == 0.5 and k["precision_on_labelled"] == 1.0
    assert len(a["daily"]) == 2 and sum(h["total"] for h in a["hourly"]) == 4
    assert a["risk_distribution"] == {"Low": 2, "Medium": 1, "High": 1}
    assert compute_analytics(df.iloc[0:0])["has_data"] is False
