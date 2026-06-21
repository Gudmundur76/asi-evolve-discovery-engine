"""Candidate molecule API endpoints.

Provides REST endpoints for listing candidates from the cognition store,
retrieving top candidates by predicted affinity, and evaluating arbitrary
SMILES strings through the trained affinity predictor.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.agents import CognitionStore
from backend.config import settings
from backend.core import AffinityPredictor, FingerprintEncoder

logger = logging.getLogger(__name__)

router = APIRouter(tags=["candidates"])

# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class EvaluateRequest(BaseModel):
    """Request body for the /evaluate endpoint."""

    smiles: str


class EvaluateResponse(BaseModel):
    """Response from evaluating a single SMILES string."""

    smiles: str
    fingerprint: List[int]
    predicted_affinity_nm: float
    is_better_than_training: bool


class CandidateItem(BaseModel):
    """A single candidate from the cognition store."""

    cycle_id: int
    smiles: str
    predicted_affinity_nm: float
    is_best_so_far: bool
    proposed_strategy: str
    lesson: str


# ---------------------------------------------------------------------------
# Lazy singletons for encoder + predictor
# ---------------------------------------------------------------------------

_encoder: Optional[FingerprintEncoder] = None
_predictor: Optional[AffinityPredictor] = None


def _get_encoder() -> FingerprintEncoder:
    """Return a shared FingerprintEncoder singleton."""
    global _encoder
    if _encoder is None:
        _encoder = FingerprintEncoder(
            radius=settings.fingerprint_radius,
            n_bits=settings.fingerprint_nbits,
        )
        logger.info("FingerprintEncoder singleton created")
    return _encoder


def _get_predictor() -> AffinityPredictor:
    """Return a shared AffinityPredictor singleton."""
    global _predictor
    if _predictor is None:
        _predictor = AffinityPredictor()
        logger.info("AffinityPredictor singleton created")
    return _predictor


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=Dict[str, Any])
async def list_candidates(
    n_recent: int = Query(default=20, ge=1, le=200),
) -> Dict[str, Any]:
    """List recent candidates from the cognition store.

    Query parameters:
        n_recent: Number of recent cycles to return (default 20).

    Returns:
        Dictionary with ``candidates`` list and ``total_cycles`` count.
    """
    try:
        # Import LoopScheduler to access the shared cognition store
        from backend.api.loop_status import _get_scheduler

        sched = _get_scheduler()
        store = sched.cognition_store
        cycles = store.cycles

        # Get the most recent N cycles
        recent_cycles = cycles[-n_recent:] if len(cycles) > n_recent else cycles

        candidates = [
            CandidateItem(
                cycle_id=c.cycle_id,
                smiles=c.new_smiles,
                predicted_affinity_nm=round(c.predicted_affinity_nm, 4),
                is_best_so_far=c.is_best_so_far,
                proposed_strategy=c.proposed_modification.get("strategy", "N/A"),
                lesson=c.lesson,
            )
            for c in reversed(recent_cycles)  # newest first
        ]

        return {
            "candidates": [c.model_dump() for c in candidates],
            "total_cycles": len(cycles),
            "best_affinity_ever": store.best_affinity_ever,
            "best_smiles_ever": store.best_smiles_ever,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("list_candidates failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to list candidates: {exc}"
        )


@router.get("/top", response_model=Dict[str, Any])
async def get_top_candidates(
    n: int = Query(default=10, ge=1, le=100),
) -> Dict[str, Any]:
    """Return the top N candidates by predicted affinity (best first).

    Query parameters:
        n: Number of top candidates to return (default 10).

    Returns:
        Dictionary with ``candidates`` list sorted by affinity ascending.
    """
    try:
        from backend.api.loop_status import _get_scheduler

        sched = _get_scheduler()
        store = sched.cognition_store
        top = store.get_top_candidates(n=n)

        candidates = [
            CandidateItem(
                cycle_id=c.cycle_id,
                smiles=c.new_smiles,
                predicted_affinity_nm=round(c.predicted_affinity_nm, 4),
                is_best_so_far=c.is_best_so_far,
                proposed_strategy=c.proposed_modification.get("strategy", "N/A"),
                lesson=c.lesson,
            )
            for c in top
        ]

        return {
            "candidates": [c.model_dump() for c in candidates],
            "target_chembl_id": store.target_chembl_id,
            "target_name": store.target_name,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_top_candidates failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get top candidates: {exc}"
        )


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_smiles(body: EvaluateRequest) -> EvaluateResponse:
    """Evaluate a SMILES string through the trained affinity predictor.

    Request body:
        smiles: Canonical SMILES string to evaluate.

    Returns:
        EvaluationResponse with fingerprint, predicted affinity, and
        comparison against the training set best.

    Raises:
        HTTPException: 400 if the SMILES cannot be parsed.
    """
    smiles = body.smiles.strip()
    if not smiles:
        raise HTTPException(status_code=400, detail="SMILES string is empty")

    try:
        encoder = _get_encoder()
        predictor = _get_predictor()

        # Encode SMILES to fingerprint
        fp = encoder.smiles_to_fp(smiles)
        if fp is None:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to encode SMILES: {smiles}",
            )

        # Convert dense numpy array to sparse list of active bits
        fp_list = fp.nonzero()[0].tolist()

        # Predict affinity
        affinity_nm = predictor.predict(fp)

        # Check if better than training
        is_better = predictor.is_better_than_training(affinity_nm)

        return EvaluateResponse(
            smiles=smiles,
            fingerprint=fp_list,
            predicted_affinity_nm=round(float(affinity_nm), 6),
            is_better_than_training=is_better,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("evaluate_smiles(%s) failed: %s", smiles, exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Evaluation failed: {exc}"
        )
