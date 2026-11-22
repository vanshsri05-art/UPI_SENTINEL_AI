"""API integration tests (FastAPI TestClient + a throw-away SQLite database).

Skipped automatically when FastAPI / SQLAlchemy / httpx are not installed or the model is untrained.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("httpx")

from ml import config  # noqa: E402

if not config.MODEL_PATH.exists():
    pytest.skip("train the model first: python run.py train", allow_module_level=True)

_tmp = tempfile.mkdtemp()
os.environ.update({
    "DATABASE_URL": f"sqlite:///{Path(_tmp, 'test.db').as_posix()}",
    "SECRET_KEY": "test-secret-key-test-secret-key-test-secret-key",
    "ADMIN_USERNAME": "testadmin", "ADMIN_PASSWORD": "Admin@12345",
    "DEMO_USERNAME": "testuser", "DEMO_PASSWORD": "User@12345",
    "FRONTEND_ORIGINS": "http://localhost:8501", "REGISTRATION_ENABLED": "true"})

from fastapi.testclient import TestClient  # noqa: E402
from backend.main import app  # noqa: E402

SPEC_EXAMPLE = {"amount": 25000, "transaction_type": "P2P", "transaction_hour": 23, "transaction_frequency": 8}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _login(client, username, password):
    r = client.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="module")
def admin(client):
    return _login(client, "testadmin", "Admin@12345")


@pytest.fixture(scope="module")
def user(client):
    return _login(client, "testuser", "User@12345")


def test_root_and_health(client):
    assert client.get("/").json()["name"] == "UPI Sentinel AI"
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_loaded"] and h["database"] == "ok"


def test_docs_available(client):
    assert client.get("/docs").status_code == 200
    assert "/predict" in client.get("/openapi.json").json()["paths"]


def test_login_failure_and_protection(client):
    assert client.post("/auth/login", json={"username": "nobody", "password": "x"}).status_code == 401
    for path in ("/transactions", "/analytics", "/model/performance"):
        assert client.get(path).status_code in (401, 403)
    assert client.post("/predict", json=SPEC_EXAMPLE).status_code in (401, 403)
    assert client.get("/analytics", headers={"Authorization": "Bearer garbage"}).status_code == 401


def test_predict_with_spec_example(client, user):
    r = client.post("/predict", json=SPEC_EXAMPLE, headers=user)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["transaction_id"].startswith("TXN")
    assert body["prediction"] in ("Genuine", "Suspicious")
    assert 0 <= body["fraud_probability"] <= 1 and body["risk_score"] == round(body["fraud_probability"] * 100)
    assert body["risk_level"] in ("Low", "Medium", "High")
    assert "sender_balance" in body["assumed_defaults"]
    assert body["top_factors"] and "is_anomaly" in body["anomaly"]


def test_predict_ranks_risky_above_safe(client, user):
    safe = client.post("/predict", headers=user, json={"amount": 450, "transaction_type": "P2M", "transaction_hour": 13,
                                                      "transaction_frequency": 2}).json()
    risky = client.post("/predict", headers=user, json={
        "amount": 48000, "transaction_type": "COLLECT_REQUEST", "transaction_hour": 2, "transaction_frequency": 9,
        "sender_balance": 50000, "receiver_balance": 0, "device_risk_score": 0.7, "is_new_device": True,
        "is_new_payee": True, "failed_attempts_24h": 3}).json()
    assert risky["fraud_probability"] > safe["fraud_probability"]
    assert risky["prediction"] == "Suspicious"


@pytest.mark.parametrize("patch", [{"amount": -5}, {"amount": 0}, {"transaction_type": "BITCOIN"},
                                   {"transaction_hour": 24}, {"transaction_frequency": -1},
                                   {"device_risk_score": 1.5}, {"unexpected": 1}])
def test_predict_validation(client, user, patch):
    assert client.post("/predict", json={**SPEC_EXAMPLE, **patch}, headers=user).status_code == 422


def test_transactions_and_detail(client, user):
    ref = client.post("/predict", json=SPEC_EXAMPLE, headers=user).json()["transaction_id"]
    r = client.get("/transactions", params={"search": ref}, headers=user)
    assert r.status_code == 200 and r.json()["total"] == 1 and r.json()["items"][0]["reference"] == ref
    assert client.get(f"/transactions/{ref}", headers=user).json()["explanation"]["factors"]
    assert client.get("/transactions/NOPE", headers=user).status_code == 404
    assert client.get("/transactions", params={"risk_level": "Extreme"}, headers=user).status_code == 422
    assert client.get("/transactions", params={"date_from": "2026-12-01", "date_to": "2026-01-01"},
                      headers=user).status_code == 422


def test_analytics_reflects_stored_data(client, user):
    a = client.get("/analytics", headers=user).json()
    n = client.get("/transactions", params={"limit": 5000}, headers=user).json()["total"]
    assert a["has_data"] and a["kpis"]["total"] == n
    assert a["kpis"]["genuine"] + a["kpis"]["suspicious"] == n


def test_model_performance_and_anomalies(client, user):
    m = client.get("/model/performance", headers=user).json()
    assert m["selected_model"] and len(m["models"]) >= 3 and m["global_importance"]["items"]
    an = client.get("/anomalies", headers=user).json()
    assert "total_anomalies" in an and "offline_comparison" in an


def test_role_based_access(client, user, admin):
    assert client.get("/admin/stats", headers=user).status_code == 403
    r = client.get("/admin/stats", headers=admin)
    assert r.status_code == 200 and r.json()["users_total"] >= 2
    assert all("password" not in u for u in r.json()["users"])


def test_users_cannot_see_each_others_transactions(client, admin):
    client.post("/auth/register", json={"username": "alice_t", "password": "Alice@1234"})
    client.post("/auth/register", json={"username": "bob_t", "password": "Bobby@1234"})
    alice, bob = _login(client, "alice_t", "Alice@1234"), _login(client, "bob_t", "Bobby@1234")
    ref = client.post("/predict", json=SPEC_EXAMPLE, headers=alice).json()["transaction_id"]
    assert client.get(f"/transactions/{ref}", headers=bob).status_code == 404
    assert client.get("/transactions", headers=bob).json()["total"] == 0
    assert client.get(f"/transactions/{ref}", headers=admin).status_code == 200


def test_registration_rules(client):
    ok = client.post("/auth/register", json={"username": "carol_t", "password": "Carol@1234"})
    assert ok.status_code == 201 and ok.json()["role"] == "user"
    assert client.post("/auth/register", json={"username": "carol_t", "password": "Carol@1234"}).status_code == 409
    assert client.post("/auth/register", json={"username": "dave_t", "password": "abcdefgh"}).status_code == 422
    assert client.post("/auth/register", json={"username": "x y", "password": "Valid@1234"}).status_code == 422
    # a client can never self-assign the admin role
    client.post("/auth/register", json={"username": "eve_t", "password": "Valid@1234", "role": "admin"})
    token = client.post("/auth/login", json={"username": "eve_t", "password": "Valid@1234"})
    assert token.status_code == 200 and token.json()["user"]["role"] == "user"


def test_cors_is_restricted(client):
    bad = client.options("/predict", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers
    good = client.options("/predict", headers={"Origin": "http://localhost:8501", "Access-Control-Request-Method": "POST"})
    assert good.headers.get("access-control-allow-origin") == "http://localhost:8501"
