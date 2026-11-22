"""Thin HTTP client used by every dashboard page. The dashboard talks to the FastAPI backend only
(never to the database directly), so API, database and UI stay properly integrated and separated."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

from ml import config as _config  # noqa: F401  (loads .env)

DEFAULT_API_URL = "http://127.0.0.1:8000"


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _detail_text(resp: requests.Response) -> str:
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text[:300]
    if isinstance(detail, list):  # FastAPI validation errors
        return "; ".join(f"{'.'.join(str(p) for p in e.get('loc', [])[1:])}: {e.get('msg')}" for e in detail)
    return str(detail)


class ApiClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or os.getenv("API_BASE_URL") or DEFAULT_API_URL).rstrip("/")

    # ---- session -----------------------------------------------------------------
    @property
    def token(self) -> Optional[str]:
        return st.session_state.get("token")

    @property
    def user(self) -> Optional[Dict[str, Any]]:
        return st.session_state.get("user")

    def logout(self) -> None:
        for key in ("token", "user"):
            st.session_state.pop(key, None)

    # ---- low level ----------------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs) -> Any:
        headers = kwargs.pop("headers", {})
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            resp = requests.request(method, f"{self.base_url}{path}", headers=headers, timeout=60, **kwargs)
        except requests.ConnectionError:
            raise ApiError(0, f"Cannot reach the API at {self.base_url}. Is the backend running? "
                              f"(python run.py api)")
        except requests.Timeout:
            raise ApiError(0, "The API took too long to respond.")
        if resp.status_code == 401 and self.token:
            self.logout()
            raise ApiError(401, "Your session expired. Please sign in again.")
        if not resp.ok:
            raise ApiError(resp.status_code, _detail_text(resp))
        return resp.json()

    # ---- endpoints -----------------------------------------------------------------
    def login(self, username: str, password: str) -> Dict[str, Any]:
        data = self._request("POST", "/auth/login", json={"username": username, "password": password})
        st.session_state["token"] = data["access_token"]
        st.session_state["user"] = data["user"]
        return data["user"]

    def register(self, username: str, password: str, email: str = "", full_name: str = "") -> Dict[str, Any]:
        body = {"username": username, "password": password}
        if email:
            body["email"] = email
        if full_name:
            body["full_name"] = full_name
        return self._request("POST", "/auth/register", json=body)

    def risk_config(self) -> Dict[str, Any]:
        return self._request("GET", "/risk/config")

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/predict", params={"source": "dashboard"}, json=payload)

    def transactions(self, **params) -> Dict[str, Any]:
        return self._request("GET", "/transactions", params={k: v for k, v in params.items() if v not in (None, [], "")})

    def transaction_detail(self, reference: str) -> Dict[str, Any]:
        return self._request("GET", f"/transactions/{reference}")

    def analytics(self) -> Dict[str, Any]:
        return self._request("GET", "/analytics")

    def anomalies(self, limit: int = 200) -> Dict[str, Any]:
        return self._request("GET", "/anomalies", params={"limit": limit})

    def model_performance(self) -> Dict[str, Any]:
        return self._request("GET", "/model/performance")

    def admin_stats(self) -> Dict[str, Any]:
        return self._request("GET", "/admin/stats")
