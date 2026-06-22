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
    """A single candidate from the cognition store.

    Quantum provenance fields are populated when quantum scoring is active.
    They are required by the ttruthdesk molecularDiscovery adapter and the
    citation.is emission bridge.
    """

    cycle_id: int
    smiles: str
    predicted_affinity_nm: float
    is_best_so_far: bool
    proposed_strategy: str
    lesson: str
    # Quantum provenance — required by ttruthdesk QUANTUM_DUAL tier
    pic50_vqe: Optional[float] = None
    quantum_hardware: Optional[str] = None
    provenance_status: Optional[str] = None
    confidence: Optional[float] = None
    citation_ids: List[str] = []


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
                pic50_vqe=round(c.pic50_vqe, 4) if c.pic50_vqe is not None else None,
                quantum_hardware=c.quantum_hardware,
                provenance_status=c.provenance_status,
                confidence=round(c.confidence, 4) if c.confidence is not None else None,
                citation_ids=list(c.citation_ids),
            )
            for c in top
        ]
        return {
            "candidates": [c.model_dump() for c in candidates],
            "target_chembl_id": store.target_chembl_id,
            "target_name": store.target_name,
            "quantum_enabled": any(c.quantum_hardware for c in top),
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


# ---------------------------------------------------------------------------
# Citation.is write-back endpoint
# ---------------------------------------------------------------------------


class CitationIdRequest(BaseModel):
    """Request body for writing a citation.is permanent URL back to a cycle."""

    citation_url: str


class CitationIdResponse(BaseModel):
    """Response after writing a citation.is URL to a cycle record."""

    cycle_id: int
    citation_url: str
    total_citation_ids: int


@router.post("/{cycle_id}/citation-id", response_model=CitationIdResponse)
async def add_citation_id(
    cycle_id: int,
    body: CitationIdRequest,
) -> CitationIdResponse:
    """Write a citation.is permanent URL back to a CycleRecord.

    Called by the ttruthdesk molecularDiscovery adapter (or any external system)
    after a candidate has been verified and assigned a permanent citation.is URL.

    The URL is appended to CycleRecord.citation_ids and also stored in
    CognitionStore.citation_registry[cycle_id] for O(1) lookup.

    The updated store is persisted to disk immediately.

    Request body:
        citation_url: Permanent citation.is URL, e.g. https://citation.is/claim/42

    Returns:
        CitationIdResponse with cycle_id, citation_url, and total_citation_ids.

    Raises:
        HTTPException 404: cycle_id not found in the cognition store.
        HTTPException 400: citation_url is empty or not a valid https URL.
    """
    url = body.citation_url.strip()
    if not url or not url.startswith("http"):
        raise HTTPException(status_code=400, detail="citation_url must be a valid https URL")

    try:
        from backend.api.loop_status import _get_scheduler

        sched = _get_scheduler()
        store = sched.cognition_store

        # Find the cycle record
        target_cycle = next(
            (c for c in store.cycles if c.cycle_id == cycle_id), None
        )
        if target_cycle is None:
            raise HTTPException(
                status_code=404,
                detail=f"cycle_id={cycle_id} not found in cognition store",
            )

        # Append the citation URL (deduplicate)
        if url not in target_cycle.citation_ids:
            target_cycle.citation_ids.append(url)

        # Update the registry index
        store.citation_registry[cycle_id] = url

        # Persist to disk
        store_path = getattr(sched, "store_path", None)
        if store_path:
            store.save(store_path)
            logger.info(
                "citation_id written: cycle_id=%d url=%s (persisted to %s)",
                cycle_id,
                url,
                store_path,
            )
        else:
            logger.warning(
                "citation_id written: cycle_id=%d url=%s (store_path unknown, not persisted)",
                cycle_id,
                url,
            )

        return CitationIdResponse(
            cycle_id=cycle_id,
            citation_url=url,
            total_citation_ids=len(target_cycle.citation_ids),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("add_citation_id(cycle_id=%d) failed: %s", cycle_id, exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to write citation ID: {exc}"
        )


# ---------------------------------------------------------------------------
# Emit-to-citation.is helper (called by the loop scheduler after each cycle)
# ---------------------------------------------------------------------------


async def emit_best_candidate_to_citation_is(
    store: "CognitionStore",  # type: ignore[name-defined]
    ttruthdesk_url: str,
    store_path: Optional[str] = None,
) -> Optional[str]:
    """Emit the current best candidate to citation.is via ttruthdesk.

    This is a fire-and-forget helper called by the loop scheduler after each
    cycle when a new best candidate is found. It POSTs to the ttruthdesk
    verify-claim endpoint and writes the returned permanent URL back to the
    Cognition Store.

    Args:
        store: The active CognitionStore instance.
        ttruthdesk_url: Base URL of the ttruthdesk deployment (citation.is backend).
        store_path: Optional path to persist the store after writing the citation ID.

    Returns:
        The permanent citation.is URL if successful, None otherwise.
    """
    import asyncio
    import aiohttp

    if not store.best_smiles_ever:
        return None

    # Find the CycleRecord for the best SMILES
    best_cycle = next(
        (
            c
            for c in reversed(store.cycles)
            if c.new_smiles == store.best_smiles_ever and c.is_best_so_far
        ),
        None,
    )
    if best_cycle is None:
        return None

    # Skip if already emitted
    if best_cycle.citation_ids:
        return best_cycle.citation_ids[0]

    # Build structured claim text
    provenance_label = (
        f"Quantum-dual verified ({best_cycle.quantum_hardware or 'multi-backend'})"
        if best_cycle.provenance_status == "QUANTUM_DUAL"
        else "Quantum-simulated (local VQE)"
        if best_cycle.provenance_status == "QUANTUM_SIM"
        else "Classical ML prediction"
    )
    pic50_text = (
        f" pIC50 (VQE) = {best_cycle.pic50_vqe:.3f}."
        if best_cycle.pic50_vqe is not None
        else ""
    )
    claim_text = (
        f"HIV-1 protease inhibitor candidate (SMILES: {best_cycle.new_smiles}) "
        f"predicted affinity {best_cycle.predicted_affinity_nm:.2f} nM. "
        f"{provenance_label}.{pic50_text} "
        f"Strategy: {best_cycle.proposed_modification.get('strategy', 'N/A')}. "
        f"Lesson: {best_cycle.lesson}"
    )

    endpoint = f"{ttruthdesk_url.rstrip('/')}/api/public/verify-claim"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                json={"claim": claim_text, "vertical": "molecular_discovery"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning(
                        "emit_best_candidate: citation.is returned HTTP %d for cycle_id=%d",
                        resp.status,
                        best_cycle.cycle_id,
                    )
                    return None

                data = await resp.json()
                claim_id = data.get("claimId")
                if claim_id is None:
                    return None

                permanent_url = f"{ttruthdesk_url.rstrip('/')}/claim/{claim_id}"

                # Write back to store
                if permanent_url not in best_cycle.citation_ids:
                    best_cycle.citation_ids.append(permanent_url)
                store.citation_registry[best_cycle.cycle_id] = permanent_url

                if store_path:
                    store.save(store_path)

                logger.info(
                    "emit_best_candidate: cycle_id=%d → %s",
                    best_cycle.cycle_id,
                    permanent_url,
                )
                return permanent_url

    except asyncio.TimeoutError:
        logger.warning("emit_best_candidate: timeout calling citation.is for cycle_id=%d", best_cycle.cycle_id)
        return None
    except Exception as exc:
        logger.error("emit_best_candidate failed for cycle_id=%d: %s", best_cycle.cycle_id, exc, exc_info=True)
        return None
