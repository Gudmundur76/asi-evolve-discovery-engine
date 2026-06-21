"""Async database session management.

Provides the SQLAlchemy async engine and session factory used by the
FastAPI dependency-injection system.
"""

from __future__ import annotations

import logging
import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.database.models import Base

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------

# Allow override via environment variable for tests / different deployments
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///data/discovery.db",
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
