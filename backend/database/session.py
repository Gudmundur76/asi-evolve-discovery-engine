"""Async database session management.

Provides the SQLAlchemy async engine and session factory used by the
FastAPI dependency-injection system.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.database.models import Base

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------

# Use MDE_DATABASE_URL (project-specific) to avoid collision with the
# sandbox-level DATABASE_URL environment variable (which points to MySQL).
# Falls back to a SQLite file in the data/ directory relative to the repo root.
_default_db_path = Path(__file__).parent.parent.parent / "data" / "hiv_protease.db"
DATABASE_URL = os.getenv(
    "MDE_DATABASE_URL",
    f"sqlite+aiosqlite:///{_default_db_path}",
)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# Lifespan helper
# ---------------------------------------------------------------------------


async def create_tables() -> None:
    """Create all ORM tables (idempotent)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ensured: %s", DATABASE_URL)


async def dispose_engine() -> None:
    """Gracefully close the connection pool."""
    await engine.dispose()
    logger.info("Database engine disposed.")


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async SQLAlchemy session for FastAPI Depends.

    Usage::

        @router.get("/items")
        async def list_items(session: AsyncSession = Depends(get_async_session)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
