"""Discovery API endpoints.

Provides REST endpoints for listing, retrieving, and downloading
molecular discovery records from the database.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.database.discovery_db import (
    count_discoveries,
    get_discovery,
    list_discoveries,
)
from backend.database.session import get_async_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["discoveries"])


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class DiscoveryResponse(BaseModel):
    """Serialized discovery record for API responses."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    candidate_id: str
    created_at: datetime
    target_chembl_id: str
    target_name: str
    smiles_hint: str
    predicted_affinity_nm: float
    training_best_nm: float
    improvement_factor: float
    docking_score: Optional[float] = None
    docking_passed: bool
    admet_druglikeness_score: Optional[float] = None
    admet_passed: bool
    confidence_score: float
    overall_passed: bool
    evidence_pdf_path: Optional[str] = None
    cycle_number: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discovery_to_response(discovery) -> DiscoveryResponse:
    """Convert a SQLAlchemy Discovery ORM object to a DiscoveryResponse.

    The ORM model stores affinity as pIC50; we convert back to nM for
    the API layer so values are human-readable.
    """
    predicted_affinity = discovery.predicted_affinity or 0.0
    # Convert pIC50 -> nM (approximate: 10^(9 - pIC50))
    predicted_nm = 10 ** (9 - predicted_affinity) if predicted_affinity > 0 else 0.0

    # Use metadata for training best (fallback to config default)
    training_best_nm = getattr(settings, "best_affinity_nM", 1000.0)
    improvement_factor = (
        training_best_nm / predicted_nm if predicted_nm > 0 else 0.0
    )

    # Cycle number from modification history length
    mod_history = discovery.modification_history or []
    cycle_number = len(mod_history) if isinstance(mod_history, list) else 0

    return DiscoveryResponse(
        id=discovery.id,
        candidate_id=discovery.candidate_id,
        created_at=discovery.created_at,
        target_chembl_id=discovery.target_chembl_id or settings.target_chembl_id,
        target_name=discovery.target_name or settings.target_name,
        smiles_hint=(discovery.smiles[:60] + "...")
        if discovery.smiles and len(discovery.smiles) > 60
        else (discovery.smiles or ""),
        predicted_affinity_nm=round(predicted_nm, 4),
        training_best_nm=round(training_best_nm, 4),
        improvement_factor=round(improvement_factor, 4),
        docking_score=discovery.docking_score,
        docking_passed=bool(discovery.docking_pass),
        admet_druglikeness_score=discovery.admet_druglikeness_score,
        admet_passed=bool(discovery.admet_overall_pass),
        confidence_score=round(float(discovery.confidence_score), 4),
        overall_passed=bool(discovery.overall_pass),
        evidence_pdf_path=discovery.evidence_pdf_path,
        cycle_number=cycle_number,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=dict)
async def list_all_discoveries(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    target_chembl_id: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_async_session),
):
    """List all discovery records with pagination.

    Query parameters:
        limit: Maximum number of records (1-500, default 50).
        offset: Number of records to skip (default 0).
        target_chembl_id: Filter by target ChEMBL ID.

    Returns:
        Dictionary with ``total`` count and ``items`` list.
    """
    try:
        total = await count_discoveries(session)
        records = await list_discoveries(
            session,
            limit=limit,
            offset=offset,
            target_chembl_id=target_chembl_id,
        )
        items = [_discovery_to_response(r) for r in records]
        return {"total": total, "items": items}
    except Exception as exc:
        logger.error("list_all_discoveries failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to list discoveries: {exc}"
        )


@router.get("/{discovery_id}", response_model=DiscoveryResponse)
async def get_discovery_by_id(
    discovery_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """Retrieve a single discovery record by ID.

    Args:
        discovery_id: Primary key of the discovery record.

    Returns:
        DiscoveryResponse for the requested record.

    Raises:
        HTTPException: 404 if the record does not exist.
    """
    try:
        record = await get_discovery(session, discovery_id)
    except Exception as exc:
        logger.error("get_discovery_by_id(%d) failed: %s", discovery_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Database query failed")

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Discovery with id={discovery_id} not found",
        )

    return _discovery_to_response(record)


@router.get("/{discovery_id}/pdf")
async def download_discovery_pdf(
    discovery_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """Download the evidence PDF for a discovery record.

    Args:
        discovery_id: Primary key of the discovery record.

    Returns:
        FileResponse with the PDF content.

    Raises:
        HTTPException: 404 if the record or PDF does not exist.
    """
    from fastapi.responses import FileResponse

    try:
        record = await get_discovery(session, discovery_id)
    except Exception as exc:
        logger.error("download_discovery_pdf(%d) query failed: %s", discovery_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Database query failed")

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Discovery with id={discovery_id} not found",
        )

    pdf_path = record.evidence_pdf_path
    if not pdf_path:
        raise HTTPException(
            status_code=404,
            detail=f"No PDF evidence available for discovery {discovery_id}",
        )

    path_obj = Path(pdf_path)
    if not path_obj.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF file not found on disk: {pdf_path}",
        )

    return FileResponse(
        path=str(path_obj),
        media_type="application/pdf",
        filename=path_obj.name,
    )
