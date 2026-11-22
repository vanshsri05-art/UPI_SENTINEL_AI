"""Framework-independent security helpers: password hashing, JWT tokens, login throttling.

* Passwords are hashed with PBKDF2-HMAC-SHA256 (stdlib), random 16-byte salt, 310k iterations.
  Plaintext passwords are never stored or logged.
* The JWT signing key is read from the SECRET_KEY environment variable - there is no default.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import jwt  # PyJWT

from ml import config as _config  # noqa: F401  (loads .env)

PBKDF2_ITERATIONS = 310_000
JWT_ALGORITHM = "HS256"


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        scheme, iterations, salt_b64, digest_b64 = stored_hash.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def validate_password_strength(password: str) -> None:
    """Raise ValueError with a readable message if the password is too weak."""
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    if len(password) > 128:
        raise ValueError("Password must be at most 128 characters long.")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise ValueError("Password must contain at least one letter and one digit.")


def get_secret_key() -> str:
    key = os.getenv("SECRET_KEY", "")
    if len(key) < 32:
        raise RuntimeError(
            "SECRET_KEY is missing or shorter than 32 characters. Run `python run.py setup` "
            "(creates .env with a random key) or set SECRET_KEY in your environment.")
    return key


def create_access_token(username: str, role: str, minutes: int | None = None) -> str:
    minutes = minutes or int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    now = datetime.now(timezone.utc)
    payload = {"sub": username, "role": role, "iat": now, "exp": now + timedelta(minutes=minutes)}
    return jwt.encode(payload, get_secret_key(), algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Dict:
    """Raises jwt.PyJWTError if the token is invalid or expired."""
    return jwt.decode(token, get_secret_key(), algorithms=[JWT_ALGORITHM])


class LoginThrottle:
    """In-memory brute-force protection: N failed logins per username per time window."""

    def __init__(self, max_attempts: int = 5, window_seconds: int = 300):
        self.max_attempts = max_attempts
        self.window = window_seconds
        self._failures: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str) -> List[float]:
        cutoff = time.time() - self.window
        recent = [t for t in self._failures.get(key, []) if t > cutoff]
        self._failures[key] = recent
        return recent

    def is_blocked(self, key: str) -> bool:
        with self._lock:
            return len(self._recent(key.lower())) >= self.max_attempts

    def record_failure(self, key: str) -> None:
        with self._lock:
            self._recent(key.lower()).append(time.time())

    def reset(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key.lower(), None)


login_throttle = LoginThrottle()
