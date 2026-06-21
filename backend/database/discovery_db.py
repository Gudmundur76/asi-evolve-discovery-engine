"""
Async CRUD operations for the Discovery table.

All functions accept an async SQLAlchemy session and return model instances
(or ``None``) as appropriate.  Connection failures are caught and logged
without propagating exceptions to the caller.

Example usage::

    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_async_engine("postgresql+asyncpg://...")
    async_session = sessionmaker(engine, class_=AsyncSession)

    async with async_session() as session:
        discovery = await create_discovery(session, {...})
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import Discovery, StatusEnum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


async def create_discovery(session: AsyncSession, data: dict[str, Any]) -> Discovery:
    """Insert a new Discovery record.

    Parameters
    ----------
    session:
        Active async SQLAlchemy session.
    data:
        Dictionary of column values.  Required keys: ``candidate_id``,
        ``smiles``.  All other keys are optional and map directly to
        :class:`Discovery` columns.

    Returns
    -------
    Discovery
        The newly created (but not yet committed) instance.
    """
    try:
        discovery = Discovery(
            candidate_id=data["candidate_id"],
            smiles=data["smiles"],
            inchi_key=data.get("inchi_key"),
            target_chembl_id=data.get("target_chembl_id"),
            target_name=data.get("target_name"),
            target_uniprot=data.get("target_uniprot"),
            predicted_affinity=data.get("predicted_affinity"),
            predicted_affinity_unit=data.get("predicted_affinity_unit", "pIC50"),
            confidence_score=data.get("confidence_score", 0.0),
            docking_score=data.get("docking_score"),
            docking_pass=data.get("docking_pass", False),
            docking_poses_path=data.get("docking_poses_path"),
            docking_best_mode=data.get("docking_best_mode"),
            docking_best_rmsd_lb=data.get("docking_best_rmsd_lb"),
            docking_best_rmsd_ub=data.get("docking_best_rmsd_ub"),
            admet_mw=data.get("admet_mw"),
            admet_logp=data.get("admet_logp"),
            admet_hbd=data.get("admet_hbd"),
            admet_hba=data.get("admet_hba"),
            admet_tpsa=data.get("admet_tpsa"),
            admet_rotatable_bonds=data.get("admet_rotatable_bonds"),
            admet_lipinski_violations=data.get("admet_lipinski_violations"),
            admet_synthetic_accessibility=data.get("admet_synthetic_accessibility"),
            admet_gi_absorption=data.get("admet_gi_absorption"),
            admet_bbb_permeable=data.get("admet_bbb_permeable"),
            admet_pgp_substrate=data.get("admet_pgp_substrate"),
            admet_druglikeness_score=data.get("admet_druglikeness_score"),
            admet_medicinal_chemistry_score=data.get("admet_medicinal_chemistry_score"),
            admet_is_druglike=data.get("admet_is_druglike"),
            admet_overall_pass=data.get("admet_overall_pass", False),
            admet_raw=data.get("admet_raw"),
            overall_pass=data.get("overall_pass", False),
            status=StatusEnum(data.get("status", "pending")),
            evidence_pdf_path=data.get("evidence_pdf_path"),
            fingerprint_hex=data.get("fingerprint_hex"),
            modification_history=data.get("modification_history", []),
        )
        session.add(discovery)
        await session.commit()
        await session.refresh(discovery)
        logger.info("Created Discovery id=%d candidate=%s", discovery.id, discovery.candidate_id)
        return discovery
    except Exception as exc:
        await session.rollback()
        logger.error("create_discovery failed: %s", exc, exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


async def get_discovery(session: AsyncSession, discovery_id: int) -> Optional[Discovery]:
    """Fetch a single Discovery by primary key.

    Returns ``None`` if the record does not exist.
    """
    try:
        result = await session.execute(
            select(Discovery).where(Discovery.id == discovery_id)
        )
        return result.scalar_one_or_none()
    except Exception as exc:
        logger.error("get_discovery(%d) failed: %s", discovery_id, exc, exc_info=True)
        return None


async def get_discovery_by_candidate_id(
    session: AsyncSession, candidate_id: str
) -> Optional[Discovery]:
    """Fetch a Discovery by its unique candidate identifier.

    Returns ``None`` if no matching record exists.
    """
    try:
        result = await session.execute(
            select(Discovery).where(Discovery.candidate_id == candidate_id)
        )
        return result.scalar_one_or_none()
    except Exception as exc:
        logger.error(
            "get_discovery_by_candidate_id(%s) failed: %s", candidate_id, exc, exc_info=True
        )
        return None


async def list_discoveries(
    session: AsyncSession,
    limit: int = 50,
    offset: int = 0,
    target_chembl_id: Optional[str] = None,
) -> list[Discovery]:
    """Return a paginated list of Discovery records.

    Parameters
    ----------
    session:
        Active async session.
    limit:
        Maximum number of records to return (default 50).
    offset:
        Number of records to skip (default 0).
    target_chembl_id:
        If provided, filter to records matching this target.

    Returns
    -------
    list[Discovery]
        Ordered by ``created_at`` descending (newest first).
    """
    try:
        stmt = select(Discovery).order_by(Discovery.created_at.desc())
        if target_chembl_id is not None:
            stmt = stmt.where(Discovery.target_chembl_id == target_chembl_id)
        stmt = stmt.offset(offset).limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())
    except Exception as exc:
        logger.error("list_discoveries failed: %s", exc, exc_info=True)
        return []


async def count_discoveries(session: AsyncSession) -> int:
    """Return the total number of Discovery records."""
    try:
        result = await session.execute(select(func.count(Discovery.id)))
        return result.scalar() or 0
    except Exception as exc:
        logger.error("count_discoveries failed: %s", exc, exc_info=True)
        return 0


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


async def update_discovery(
    session: AsyncSession, discovery_id: int, updates: dict[str, Any]
) -> Optional[Discovery]:
    """Apply partial updates to an existing Discovery record.

    Parameters
    ----------
    session:
        Active async session.
    discovery_id:
        Primary key of the record to update.
    updates:
        Dictionary of column-name -> new-value pairs.  ``updated_at``
        is refreshed automatically.

    Returns
    -------
    Discovery or None
        The updated instance, or ``None`` if the record was not found.
    """
    try:
        # Refresh updated_at automatically
        updates["updated_at"] = datetime.now(timezone.utc)

        result = await session.execute(
            update(Discovery)
            .where(Discovery.id == discovery_id)
            .values(**updates)
            .execution_options(synchronize_session="fetch")
        )
        if result.rowcount == 0:
            logger.warning("update_discovery(%d): no matching record", discovery_id)
            return None

        await session.commit()

        # Re-fetch to return fresh state
        discovery = await get_discovery(session, discovery_id)
        if discovery:
            logger.info("Updated Discovery id=%d", discovery_id)
        return discovery
    except Exception as exc:
        await session.rollback()
        logger.error("update_discovery(%d) failed: %s", discovery_id, exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


async def delete_discovery(session: AsyncSession, discovery_id: int) -> bool:
    """Remove a Discovery record by primary key.

    Parameters
    ----------
    session:
        Active async session.
    discovery_id:
        Primary key of the record to delete.

    Returns
    -------
    bool
        ``True`` if a record was deleted, ``False`` otherwise.
    """
    try:
        discovery = await get_discovery(session, discovery_id)
        if discovery is None:
            logger.warning("delete_discovery(%d): record not found", discovery_id)
            return False

        await session.delete(discovery)
        await session.commit()
        logger.info("Deleted Discovery id=%d", discovery_id)
        return True
    except Exception as exc:
        await session.rollback()
        logger.error("delete_discovery(%d) failed: %s", discovery_id, exc, exc_info=True)
        return False
