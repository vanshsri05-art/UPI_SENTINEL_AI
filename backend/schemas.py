"""Pydantic v2 request / response schemas (input validation lives here)."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

TransactionType = Literal["P2P", "P2M", "BILL_PAYMENT", "QR_PAYMENT", "COLLECT_REQUEST",
                          "CASH_OUT", "CASH_IN"]
RiskLevel = Literal["Low", "Medium", "High"]


class TransactionInput(BaseModel):
    """Features of ONE payment to analyse. Optional fields fall back to typical low-risk values
    (reported back as `assumed_defaults`)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "amount": 25000, "transaction_type": "P2P", "transaction_hour": 23,
            "transaction_frequency": 8, "sender_balance": 40000, "receiver_balance": 0,
            "device_risk_score": 0.6, "is_new_device": True, "is_new_payee": True,
            "failed_attempts_24h": 2}},
    )

    amount: float = Field(..., gt=0, le=1_000_000, allow_inf_nan=False, description="Amount in INR")
    transaction_type: TransactionType
    transaction_hour: int = Field(..., ge=0, le=23)
    transaction_frequency: int = Field(..., ge=0, le=500, description="Sender transactions in last 24h")
    sender_balance: Optional[float] = Field(None, ge=0, le=1e9, allow_inf_nan=False)
    receiver_balance: Optional[float] = Field(None, ge=0, le=1e9, allow_inf_nan=False)
    device_risk_score: Optional[float] = Field(None, ge=0, le=1, allow_inf_nan=False)
    is_new_device: Optional[bool] = None
    is_new_payee: Optional[bool] = None
    failed_attempts_24h: Optional[int] = Field(None, ge=0, le=50)

    @field_validator("transaction_type", mode="before")
    @classmethod
    def _normalise_type(cls, v: Any) -> Any:
        return v.strip().upper() if isinstance(v, str) else v


class FactorOut(BaseModel):
    feature: str
    label: str
    value: str
    contribution: float
    relative_impact: float
    direction: str
    text: str


class AnomalyOut(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    note: str = ("Unsupervised Isolation Forest signal: 'unusual', not 'fraud'. "
                 "Shown separately from the fraud prediction.")


class PredictionResponse(BaseModel):
    transaction_id: str
    prediction: Literal["Genuine", "Suspicious"]
    fraud_probability: float
    risk_score: int
    risk_level: RiskLevel
    anomaly: AnomalyOut
    model_name: str
    explanation_method: str
    summary: str
    top_factors: List[FactorOut]
    assumed_defaults: List[str]
    timestamp: str


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=32)
    password: str = Field(..., min_length=1, max_length=128)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(..., min_length=8, max_length=128)
    email: Optional[str] = Field(None, max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    full_name: Optional[str] = Field(None, max_length=80)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: str
    is_active: bool
    created_at: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    user: UserOut


class TransactionListOut(BaseModel):
    total: int
    limit: int
    offset: int
    scope: Literal["all", "own"]
    items: List[Dict[str, Any]]
