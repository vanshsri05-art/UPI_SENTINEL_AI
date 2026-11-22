"""Database-facing business logic shared by the API and the CLI (seed / bootstrap)."""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.database import DATABASE_URL, engine
from backend.models_db import ModelPerformance, PredictionRecord, Transaction, User, utcnow
from backend.security import hash_password, validate_password_strength
from ml.config import RANDOM_STATE, ROOT_DIR
from ml.preprocessing import RAW_FEATURE_COLUMNS, TARGET_COLUMN, clean_dataset, load_dataset, split_data

log = logging.getLogger("upi_sentinel")

DF_COLUMNS = ["reference", "user_id", "created_at", "amount", "transaction_type", "transaction_hour",
              "transaction_frequency", "sender_balance", "receiver_balance", "device_risk_score",
              "is_new_device", "is_new_payee", "failed_attempts_24h", "actual_label", "source",
              "prediction", "fraud_probability", "risk_score", "risk_level", "is_anomaly",
              "anomaly_score", "model_name"]


def new_reference() -> str:
    return "TXN" + utcnow().strftime("%y%m%d") + uuid.uuid4().hex[:8].upper()


# ----------------------------------------------------------------------------- users
def create_user(db: Session, username: str, password: str, role: str = "user",
                email: Optional[str] = None, full_name: Optional[str] = None) -> User:
    validate_password_strength(password)
    user = User(username=username.strip().lower(), password_hash=hash_password(password), role=role,
                email=email, full_name=full_name)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.scalar(select(User).where(User.username == username.strip().lower()))


def bootstrap_users(db: Session) -> List[str]:
    """Create the admin and demo users from environment variables if they do not exist yet."""
    created = []
    for role, u_env, p_env in (("admin", "ADMIN_USERNAME", "ADMIN_PASSWORD"),
                               ("user", "DEMO_USERNAME", "DEMO_PASSWORD")):
        username, password = os.getenv(u_env), os.getenv(p_env)
        if not username or not password:
            continue
        if get_user_by_username(db, username) is None:
            try:
                create_user(db, username, password, role=role, full_name=f"Demo {role}")
                created.append(username)
            except ValueError as exc:
                log.warning("Skipping %s: %s", u_env, exc)
    return created


# ----------------------------------------------------------------------------- predictions
def store_prediction(db: Session, user: Optional[User], record: Dict[str, Any], result: Dict[str, Any],
                     source: str = "api", actual_label: Optional[int] = None,
                     created_at: Optional[datetime] = None) -> Transaction:
    ts = created_at or utcnow()
    explanation = None
    if "top_factors" in result:
        explanation = json.dumps({"method": result.get("explanation_method"),
                                  "summary": result.get("summary"), "factors": result["top_factors"]})
    txn = Transaction(
        reference=new_reference(), user_id=user.id if user else None, amount=float(record["amount"]),
        transaction_type=record["transaction_type"], transaction_hour=int(record["transaction_hour"]),
        transaction_frequency=int(record["transaction_frequency"]),
        sender_balance=float(record["sender_balance"]), receiver_balance=float(record["receiver_balance"]),
        device_risk_score=float(record["device_risk_score"]), is_new_device=bool(record["is_new_device"]),
        is_new_payee=bool(record["is_new_payee"]), failed_attempts_24h=int(record["failed_attempts_24h"]),
        actual_label=actual_label, source=source, created_at=ts)
    txn.prediction = PredictionRecord(
        model_name=result["model_name"], prediction=result["prediction"],
        fraud_probability=float(result["fraud_probability"]), risk_score=int(result["risk_score"]),
        risk_level=result["risk_level"], is_anomaly=bool(result["is_anomaly"]),
        anomaly_score=float(result["anomaly_score"]), explanation_json=explanation, created_at=ts)
    db.add(txn)
    db.commit()
    db.refresh(txn)
    return txn


def query_transactions(db: Session, user_id: Optional[int] = None, search: Optional[str] = None,
                       risk_levels: Optional[Sequence[str]] = None,
                       transaction_types: Optional[Sequence[str]] = None,
                       prediction: Optional[str] = None, date_from: Optional[date] = None,
                       date_to: Optional[date] = None, anomaly_only: bool = False,
                       limit: Optional[int] = None, offset: int = 0) -> Tuple[pd.DataFrame, int]:
    """Filtered, newest-first transactions joined with their predictions. All values are bound
    parameters (SQLAlchemy), so user input never reaches the SQL text."""
    conds = []
    if user_id is not None:
        conds.append(Transaction.user_id == user_id)
    if search:
        term = f"%{search.strip()[:60]}%"
        conds.append(Transaction.reference.ilike(term) | Transaction.transaction_type.ilike(term))
    if risk_levels:
        conds.append(PredictionRecord.risk_level.in_(list(risk_levels)))
    if transaction_types:
        conds.append(Transaction.transaction_type.in_(list(transaction_types)))
    if prediction:
        conds.append(PredictionRecord.prediction == prediction)
    if anomaly_only:
        conds.append(PredictionRecord.is_anomaly.is_(True))
    if date_from:
        conds.append(Transaction.created_at >= datetime.combine(date_from, time.min))
    if date_to:
        conds.append(Transaction.created_at <= datetime.combine(date_to, time.max))

    base = select(Transaction, PredictionRecord).join(
        PredictionRecord, PredictionRecord.transaction_id == Transaction.id).where(*conds)
    total = db.scalar(
        select(func.count(Transaction.id)).select_from(Transaction)
        .join(PredictionRecord, PredictionRecord.transaction_id == Transaction.id).where(*conds)) or 0
    stmt = base.order_by(Transaction.created_at.desc(), Transaction.id.desc()).offset(offset)
    if limit is not None:
        stmt = stmt.limit(limit)

    rows = []
    for t, p in db.execute(stmt).all():
        rows.append({
            "reference": t.reference, "user_id": t.user_id, "created_at": t.created_at,
            "amount": t.amount, "transaction_type": t.transaction_type,
            "transaction_hour": t.transaction_hour, "transaction_frequency": t.transaction_frequency,
            "sender_balance": t.sender_balance, "receiver_balance": t.receiver_balance,
            "device_risk_score": t.device_risk_score, "is_new_device": t.is_new_device,
            "is_new_payee": t.is_new_payee, "failed_attempts_24h": t.failed_attempts_24h,
            "actual_label": t.actual_label, "source": t.source, "prediction": p.prediction,
            "fraud_probability": p.fraud_probability, "risk_score": p.risk_score,
            "risk_level": p.risk_level, "is_anomaly": p.is_anomaly, "anomaly_score": p.anomaly_score,
            "model_name": p.model_name})
    return pd.DataFrame(rows, columns=DF_COLUMNS), int(total)


def dataframe_to_items(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """JSON-safe list of dicts (datetimes -> ISO strings, numpy -> python)."""
    items = []
    for rec in df.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if isinstance(v, (pd.Timestamp, datetime)):
                v = v.isoformat(timespec="seconds")
            elif isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = None if np.isnan(v) else float(v)
            elif isinstance(v, (np.bool_,)):
                v = bool(v)
            elif v is not None and isinstance(v, float) and np.isnan(v):
                v = None
            clean[k] = v
        items.append(clean)
    return items


def get_transaction_detail(db: Session, reference: str, user_id: Optional[int]) -> Optional[Dict[str, Any]]:
    stmt = select(Transaction).where(Transaction.reference == reference)
    if user_id is not None:
        stmt = stmt.where(Transaction.user_id == user_id)
    t = db.scalar(stmt)
    if t is None or t.prediction is None:
        return None
    p = t.prediction
    return {
        "reference": t.reference, "created_at": t.created_at.isoformat(timespec="seconds"),
        "features": {c: getattr(t, c) for c in RAW_FEATURE_COLUMNS},
        "actual_label": t.actual_label, "source": t.source,
        "prediction": p.prediction, "fraud_probability": p.fraud_probability,
        "risk_score": p.risk_score, "risk_level": p.risk_level, "is_anomaly": p.is_anomaly,
        "anomaly_score": p.anomaly_score, "model_name": p.model_name,
        "explanation": json.loads(p.explanation_json) if p.explanation_json else None,
    }


# ----------------------------------------------------------------------------- model performance
def sync_model_performance(db: Session, metadata: Dict[str, Any]) -> None:
    trained_at = datetime.fromisoformat(metadata["trained_at"]).replace(tzinfo=None)
    latest = db.scalar(select(func.max(ModelPerformance.trained_at)))
    if latest == trained_at:
        return
    db.query(ModelPerformance).delete()
    for m in metadata["models"]:
        t = m["test"]
        db.add(ModelPerformance(
            model_name=m["name"], accuracy=t["accuracy"], precision=t["precision"], recall=t["recall"],
            f1=t["f1"], roc_auc=t["roc_auc"], pr_auc=t["pr_auc"], threshold=m["threshold"],
            is_selected=bool(m["is_selected"]), trained_at=trained_at))
    db.commit()


# ----------------------------------------------------------------------------- seeding
def seed_demo_data(db: Session, predictor, n: int = 1500, force: bool = False) -> int:
    """Score a sample of the HELD-OUT TEST split with the real model and store it.

    Timestamps are spread over the last 30 days (matching each row's transaction_hour) so the
    dashboards have a time axis. The rows themselves are synthetic - this is demo data, labelled
    source='seed'. Rows are owned by the demo user so the regular-user dashboard is populated.
    """
    if not force and db.scalar(select(func.count()).select_from(Transaction).where(
            Transaction.source == "seed")):
        return 0
    owner = get_user_by_username(db, os.getenv("DEMO_USERNAME", "")) if os.getenv("DEMO_USERNAME") else None
    if owner is None:
        owner = db.scalar(select(User).where(User.role == "admin"))

    df, _ = clean_dataset(load_dataset("synthetic"))
    _, _, test = split_data(df)
    sample = test.sample(n=min(n, len(test)), random_state=RANDOM_STATE).reset_index(drop=True)
    records = sample[RAW_FEATURE_COLUMNS].to_dict(orient="records")
    results = predictor.predict_records(records, explain=True)

    rng = np.random.default_rng(RANDOM_STATE)
    today = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    objects = []
    for rec, res, label in zip(records, results, sample[TARGET_COLUMN].tolist()):
        ts = today - timedelta(days=int(rng.integers(0, 30))) + timedelta(
            hours=int(rec["transaction_hour"]), minutes=int(rng.integers(0, 60)))
        if ts > utcnow():
            ts -= timedelta(days=1)
        txn = Transaction(
            reference=new_reference(), user_id=owner.id if owner else None, amount=float(rec["amount"]),
            transaction_type=rec["transaction_type"], transaction_hour=int(rec["transaction_hour"]),
            transaction_frequency=int(rec["transaction_frequency"]),
            sender_balance=float(rec["sender_balance"]), receiver_balance=float(rec["receiver_balance"]),
            device_risk_score=float(rec["device_risk_score"]), is_new_device=bool(rec["is_new_device"]),
            is_new_payee=bool(rec["is_new_payee"]), failed_attempts_24h=int(rec["failed_attempts_24h"]),
            actual_label=int(label), source="seed", created_at=ts)
        txn.prediction = PredictionRecord(
            model_name=res["model_name"], prediction=res["prediction"],
            fraud_probability=res["fraud_probability"], risk_score=res["risk_score"],
            risk_level=res["risk_level"], is_anomaly=res["is_anomaly"], anomaly_score=res["anomaly_score"],
            explanation_json=json.dumps({"method": res["explanation_method"], "summary": res["summary"],
                                         "factors": res["top_factors"]}), created_at=ts)
        objects.append(txn)
    db.add_all(objects)
    db.commit()
    return len(objects)


# ----------------------------------------------------------------------------- admin stats
def database_statistics(db: Session) -> Dict[str, Any]:
    counts = {
        "users": db.scalar(select(func.count()).select_from(User)) or 0,
        "transactions": db.scalar(select(func.count()).select_from(Transaction)) or 0,
        "prediction_history": db.scalar(select(func.count()).select_from(PredictionRecord)) or 0,
        "model_performance": db.scalar(select(func.count()).select_from(ModelPerformance)) or 0,
    }
    size_kb = None
    if DATABASE_URL.startswith("sqlite"):
        path = engine.url.database
        if path and os.path.exists(path):
            size_kb = round(os.path.getsize(path) / 1024, 1)
    last = db.scalar(select(func.max(PredictionRecord.created_at)))
    return {"dialect": engine.dialect.name, "table_rows": counts, "size_kb": size_kb,
            "last_prediction_at": last.isoformat(timespec="seconds") if last else None}
