from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from w8t.config import settings

_sqlite = settings.database_url.startswith("sqlite")
# pool_pre_ping: Neon's free tier suspends the database when idle, silently killing pooled
# connections; the ping replaces a dead one instead of failing the page.
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if _sqlite else {},
    pool_pre_ping=not _sqlite,
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
