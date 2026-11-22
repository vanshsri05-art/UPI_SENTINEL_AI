"""All REST endpoints. Interactive docs: /docs (Swagger UI) and /redoc."""
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from backend import analytics, services
from backend.auth import authenticate_user, get_current_user, require_admin
from backend.database import get_db
from backend.models_db import ModelPerformance, User
from backend.schemas import (LoginRequest, PredictionResponse, RegisterRequest, RiskLevel,
                             TokenResponse, TransactionInput, TransactionListOut, TransactionType,
                             UserOut)
from backend.security import create_access_token, validate_password_strength
from ml import config
from ml.predict import FraudPredictor, ModelNotFoundError, get_predictor
from ml.preprocessing import OPTIONAL_DEFAULTS

router = APIRouter()
APP_VERSION = "1.0.0"


def _user_out(u: User) -> UserOut:
    return UserOut(id=u.id, username=u.username, email=u.email, full_name=u.full_name, role=u.role,
                   is_active=u.is_active, created_at=u.created_at.isoformat(timespec="seconds"))


def get_model() -> FraudPredictor:
    try:
        return get_predictor()
    except ModelNotFoundError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))


def _scope_user_id(user: User) -> Optional[int]:
    """Admins see every record; regular users only their own."""
    return None if user.role == "admin" else user.id


# ------------------------------------------------------------------ meta
@router.get("/", tags=["Meta"], summary="API information")
def root() -> Dict[str, Any]:
    return {"name": "UPI Sentinel AI", "tagline": "Intelligent Digital Payment Fraud Detection & Risk Intelligence",
            "version": APP_VERSION, "docs": "/docs", "health": "/health",
            "disclaimer": "Academic decision-support system trained on synthetic data. "
                          "It does not connect to banks and never blocks or initiates payments."}


@router.get("/health", tags=["Meta"], summary="Service health")
def health(request: Request, db: Session = Depends(get_db)) -> Dict[str, Any]:
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    model_error = getattr(request.app.state, "model_error", None)
    try:
        get_predictor()
        model_error = None
    except ModelNotFoundError as exc:
        model_error = str(exc)
    ok = db_ok and model_error is None
    return {"status": "ok" if ok else "degraded", "database": "ok" if db_ok else "error",
            "model_loaded": model_error is None, "model_error": model_error,
            "time": datetime.utcnow().isoformat(timespec="seconds") + "Z"}


@router.get("/risk/config", tags=["Meta"], summary="Project-defined risk bands")
def risk_config(_: User = Depends(get_current_user)) -> Dict[str, Any]:
    return {"low": [0, config.RISK_LOW_MAX], "medium": [config.RISK_LOW_MAX + 1, config.RISK_MEDIUM_MAX],
            "high": [config.RISK_MEDIUM_MAX + 1, 100],
            "note": "Project-defined thresholds (env RISK_LOW_MAX / RISK_MEDIUM_MAX), "
                    "not an official banking or NPCI standard."}


# ------------------------------------------------------------------ auth
@router.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
def login(body: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = authenticate_user(db, body.username, body.password)
    minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    return TokenResponse(access_token=create_access_token(user.username, user.role, minutes),
                         expires_in_minutes=minutes, user=_user_out(user))


@router.post("/auth/register", response_model=UserOut, status_code=status.HTTP_201_CREATED, tags=["Auth"],
             summary="Create a regular (non-admin) account")
def register(body: RegisterRequest, db: Session = Depends(get_db)) -> UserOut:
    if os.getenv("REGISTRATION_ENABLED", "true").lower() != "true":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Registration is disabled.")
    try:
        validate_password_strength(body.password)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    if services.get_user_by_username(db, body.username):
        raise HTTPException(status.HTTP_409_CONFLICT, "Username already exists.")
    user = services.create_user(db, body.username, body.password, role="user",
                                email=body.email, full_name=body.full_name)
    return _user_out(user)


@router.get("/auth/me", response_model=UserOut, tags=["Auth"])
def me(user: User = Depends(get_current_user)) -> UserOut:
    return _user_out(user)


# ------------------------------------------------------------------ prediction
@router.post("/predict", response_model=PredictionResponse, tags=["Fraud detection"],
             summary="Analyse one transaction with the trained model")
def predict(payload: TransactionInput, db: Session = Depends(get_db),
            user: User = Depends(get_current_user), predictor: FraudPredictor = Depends(get_model),
            source: str = Query("api", pattern="^(api|dashboard)$")) -> PredictionResponse:
    assumed = [k for k in OPTIONAL_DEFAULTS if getattr(payload, k) is None]
    record = {k: v for k, v in payload.model_dump().items() if v is not None}
    for flag in ("is_new_device", "is_new_payee"):
        if flag in record:
            record[flag] = int(record[flag])
    full_record = {**OPTIONAL_DEFAULTS, **record}
    try:
        result = predictor.predict_one(full_record, explain=True)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    txn = services.store_prediction(db, user, full_record, result, source=source)
    return PredictionResponse(
        transaction_id=txn.reference, prediction=result["prediction"],
        fraud_probability=result["fraud_probability"], risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        anomaly={"is_anomaly": result["is_anomaly"], "anomaly_score": result["anomaly_score"]},
        model_name=result["model_name"], explanation_method=result["explanation_method"],
        summary=result["summary"], top_factors=result["top_factors"], assumed_defaults=assumed,
        timestamp=txn.created_at.isoformat(timespec="seconds") + "Z")


# ------------------------------------------------------------------ transactions
@router.get("/transactions", response_model=TransactionListOut, tags=["Transactions"],
            summary="Transaction history (admin: all, user: own)")
def list_transactions(
        search: Optional[str] = Query(None, max_length=60),
        risk_level: Optional[List[RiskLevel]] = Query(None),
        transaction_type: Optional[List[TransactionType]] = Query(None),
        prediction: Optional[str] = Query(None, pattern="^(Genuine|Suspicious)$"),
        date_from: Optional[date] = None, date_to: Optional[date] = None,
        anomaly_only: bool = False, limit: int = Query(500, ge=1, le=5000), offset: int = Query(0, ge=0),
        db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> TransactionListOut:
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "date_from must not be after date_to.")
    df, total = services.query_transactions(
        db, user_id=_scope_user_id(user), search=search, risk_levels=risk_level,
        transaction_types=transaction_type, prediction=prediction, date_from=date_from, date_to=date_to,
        anomaly_only=anomaly_only, limit=limit, offset=offset)
    return TransactionListOut(total=total, limit=limit, offset=offset,
                              scope="all" if user.role == "admin" else "own",
                              items=services.dataframe_to_items(df))


@router.get("/transactions/{reference}", tags=["Transactions"], summary="One transaction with its explanation")
def transaction_detail(reference: str, db: Session = Depends(get_db),
                       user: User = Depends(get_current_user)) -> Dict[str, Any]:
    detail = services.get_transaction_detail(db, reference[:40], _scope_user_id(user))
    if detail is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Transaction not found.")
    return detail


# ------------------------------------------------------------------ analytics
@router.get("/analytics", tags=["Analytics"], summary="Dashboard analytics computed from stored data")
def get_analytics(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> Dict[str, Any]:
    df, _ = services.query_transactions(db, user_id=_scope_user_id(user))
    out = analytics.compute_analytics(df)
    out["scope"] = "all" if user.role == "admin" else "own"
    out["risk_thresholds"] = {"low_max": config.RISK_LOW_MAX, "medium_max": config.RISK_MEDIUM_MAX}
    return out


@router.get("/anomalies", tags=["Analytics"],
            summary="Isolation-Forest anomalies (reported separately from fraud predictions)")
def get_anomalies(limit: int = Query(200, ge=1, le=2000), db: Session = Depends(get_db),
                  user: User = Depends(get_current_user),
                  predictor: FraudPredictor = Depends(get_model)) -> Dict[str, Any]:
    df, total = services.query_transactions(db, user_id=_scope_user_id(user), anomaly_only=True, limit=limit)
    both = int((df["prediction"] == "Suspicious").sum()) if len(df) else 0
    return {"total_anomalies": total, "shown": len(df), "also_flagged_suspicious": both,
            "anomaly_only": len(df) - both,
            "offline_comparison": predictor.metadata["anomaly"],
            "items": services.dataframe_to_items(df),
            "note": "An anomaly is an unusual transaction, not a confirmed fraud."}


# ------------------------------------------------------------------ model
@router.get("/model/performance", tags=["Model"], summary="Training results, curves and feature importance")
def model_performance(_: User = Depends(get_current_user),
                      predictor: FraudPredictor = Depends(get_model)) -> Dict[str, Any]:
    m = predictor.metadata
    return {k: m[k] for k in ("trained_at", "dataset", "split", "selected_model", "decision_threshold",
                              "selection_criterion", "imbalance_handling", "models", "global_importance",
                              "anomaly", "risk_thresholds", "shap_available", "xgboost_available")
            } | {"explainer_method": predictor.explainer.method}


# ------------------------------------------------------------------ admin
@router.get("/admin/stats", tags=["Admin"], summary="Platform statistics (admin only)")
def admin_stats(db: Session = Depends(get_db), _: User = Depends(require_admin)) -> Dict[str, Any]:
    df, _total = services.query_transactions(db)
    users = db.scalars(select(User).order_by(User.created_at)).all()
    models = db.scalars(select(ModelPerformance).order_by(ModelPerformance.f1.desc())).all()
    suspicious = df[df["prediction"] == "Suspicious"] if len(df) else df
    return {
        "users_total": len(users),
        "users_by_role": {r: sum(1 for u in users if u.role == r) for r in ("admin", "user")},
        "users": [_user_out(u).model_dump() for u in users],
        "transactions_total": int(len(df)),
        "suspicious_total": int(len(suspicious)),
        "suspicious_recent": services.dataframe_to_items(suspicious.head(25)),
        "analytics": analytics.compute_analytics(df),
        "model_performance": [{"model_name": m.model_name, "accuracy": m.accuracy, "precision": m.precision,
                               "recall": m.recall, "f1": m.f1, "roc_auc": m.roc_auc, "pr_auc": m.pr_auc,
                               "threshold": m.threshold, "is_selected": m.is_selected} for m in models],
        "database": services.database_statistics(db),
    }
