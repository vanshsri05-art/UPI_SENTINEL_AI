"""ORM tables: users, transactions, prediction history, model performance.

No UPI PINs, OTPs, real UPI IDs or bank credentials are stored - only the numeric features the
model needs plus an opaque reference ID.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


def utcnow() -> datetime:
    """Naive UTC timestamp (SQLite has no timezone type)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    email: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(10), default="user")  # 'admin' | 'user'
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    transactions: Mapped[list["Transaction"]] = relationship(back_populates="user")


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    reference: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), index=True, nullable=True)
    amount: Mapped[float] = mapped_column(Float)
    transaction_type: Mapped[str] = mapped_column(String(24), index=True)
    transaction_hour: Mapped[int] = mapped_column(Integer)
    transaction_frequency: Mapped[int] = mapped_column(Integer)
    sender_balance: Mapped[float] = mapped_column(Float)
    receiver_balance: Mapped[float] = mapped_column(Float)
    device_risk_score: Mapped[float] = mapped_column(Float)
    is_new_device: Mapped[bool] = mapped_column(Boolean)
    is_new_payee: Mapped[bool] = mapped_column(Boolean)
    failed_attempts_24h: Mapped[int] = mapped_column(Integer)
    actual_label: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # known only for seeded data
    source: Mapped[str] = mapped_column(String(16), default="api")  # 'api' | 'dashboard' | 'seed'
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    user: Mapped[Optional[User]] = relationship(back_populates="transactions")
    prediction: Mapped[Optional["PredictionRecord"]] = relationship(
        back_populates="transaction", uselist=False, cascade="all, delete-orphan")


class PredictionRecord(Base):
    __tablename__ = "prediction_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id"), unique=True, index=True)
    model_name: Mapped[str] = mapped_column(String(40))
    prediction: Mapped[str] = mapped_column(String(12), index=True)  # 'Genuine' | 'Suspicious'
    fraud_probability: Mapped[float] = mapped_column(Float)
    risk_score: Mapped[int] = mapped_column(Integer, index=True)
    risk_level: Mapped[str] = mapped_column(String(8), index=True)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    anomaly_score: Mapped[float] = mapped_column(Float, default=0.0)
    explanation_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    transaction: Mapped[Transaction] = relationship(back_populates="prediction")


class ModelPerformance(Base):
    __tablename__ = "model_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(40))
    accuracy: Mapped[float] = mapped_column(Float)
    precision: Mapped[float] = mapped_column(Float)
    recall: Mapped[float] = mapped_column(Float)
    f1: Mapped[float] = mapped_column(Float)
    roc_auc: Mapped[float] = mapped_column(Float)
    pr_auc: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    is_selected: Mapped[bool] = mapped_column(Boolean, default=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
