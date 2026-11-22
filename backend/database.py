"""Database engine and session management (SQLAlchemy 2.x).

SQLite is the default. For MySQL set, e.g.:
    DATABASE_URL=mysql+pymysql://user:password@localhost:3306/upi_sentinel
and `pip install pymysql`. No code change is required.
"""
from __future__ import annotations

import os
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ml.config import ROOT_DIR

DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{(ROOT_DIR / 'upi_sentinel.db').as_posix()}"
_is_sqlite = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    pool_pre_ping=True,
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from backend import models_db  # noqa: F401  (registers the tables)

    Base.metadata.create_all(bind=engine)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
