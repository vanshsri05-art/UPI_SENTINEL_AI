"""Authentication / authorisation: FastAPI dependencies for bearer-token auth and admin-only routes."""
from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models_db import User
from backend.security import decode_access_token, login_throttle, verify_password
from backend.services import get_user_by_username

bearer_scheme = HTTPBearer(auto_error=False, description="Paste the access_token from POST /auth/login")

# Pre-computed hash used to keep timing similar when the username does not exist.
_DUMMY_HASH = "pbkdf2_sha256$310000$AAAAAAAAAAAAAAAAAAAAAA==$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="


def authenticate_user(db: Session, username: str, password: str) -> User:
    if login_throttle.is_blocked(username):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            "Too many failed login attempts. Try again in a few minutes.")
    user = get_user_by_username(db, username)
    ok = verify_password(password, user.password_hash if user else _DUMMY_HASH)
    if not user or not ok or not user.is_active:
        login_throttle.record_failure(username)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect username or password.")
    login_throttle.reset(username)
    return user


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
                     db: Session = Depends(get_db)) -> User:
    unauthorised = HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated.",
                                 headers={"WWW-Authenticate": "Bearer"})
    if credentials is None:
        raise unauthorised
    try:
        payload = decode_access_token(credentials.credentials)
    except jwt.PyJWTError:
        raise unauthorised
    user = get_user_by_username(db, str(payload.get("sub", "")))
    if user is None or not user.is_active:
        raise unauthorised
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Administrator access required.")
    return user
