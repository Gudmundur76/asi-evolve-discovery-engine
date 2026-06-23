"""Loop control API endpoints.

Provides REST endpoints for starting, stopping, and monitoring the
three-agent molecular discovery optimization loop.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from backend.agents import LoopScheduler
from backend.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["loop"])

# ---------------------------------------------------------------------------
# Module-level singleton (lazily initialized)
# ---------------------------------------------------------------------------

_scheduler: Optional[LoopScheduler] = None
_continuous_task: Optional[asyncio.Task] = None


def _get_scheduler() -> LoopScheduler:
    """Return the shared LoopScheduler singleton, creating it if needed.

    Lazy initialisation avoids heavy imports / model loading at import
    time, which keeps uvicorn startup fast and prevents issues when
    the module is imported by tools that don't need the scheduler.
    """
    global _scheduler
    if _scheduler is None:
        try:
            _scheduler = LoopScheduler()
            logger.info("LoopScheduler singleton created")
        except Exception as exc:
            logger.error("Failed to create LoopScheduler: %s", exc, exc_info=True)
            detail = str(exc)
            # Model not yet trained (cold-start / bootstrap in progress) → 503 Retry
            if "Model file not found" in detail or "model.pkl" in detail:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Model not ready yet — auto-bootstrap training is in progress. "
                        "Check GET /health for model_ready status and retry in ~60 s."
                    ),
                )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize LoopScheduler: {exc}",
            )
    return _scheduler


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/status")
async def get_loop_status() -> Dict[str, Any]:
    """Get the current status of the optimization loop.

    Returns:
        Dictionary with running state, cycle count, best affinity,
        timestamps, and target information.
    """
    sched = _get_scheduler()
    status = sched.get_status()
    return {
        "running": status.get("running", False),
        "cycle_count": status.get("cycle_count", 0),
        "current_best_affinity": status.get("current_best_affinity", float("inf")),
        "last_cycle_time": status.get("last_cycle_time"),
        "next_cycle_time": status.get("next_cycle_time"),
        "target_chembl_id": status.get("target_chembl_id", settings.target_chembl_id),
        "target_name": status.get("target", settings.target_name),
    }


@router.post("/start")
async def start_loop() -> Dict[str, str]:
    """Start the continuous optimization loop in the background.

    Returns:
        Confirmation message with the target being optimised.
    """
    global _continuous_task
    sched = _get_scheduler()

    if sched.running:
        return {"message": "Loop is already running", "status": "already_running"}

    try:
        sched.start()

        # Launch the continuous loop as a background asyncio task
        _continuous_task = asyncio.create_task(
            _run_continuous_wrapper(), name="loop_scheduler_continuous"
        )

        logger.info("Continuous loop started for target %s", settings.target_chembl_id)
        return {
            "message": "Continuous loop started",
            "target_chembl_id": settings.target_chembl_id,
            "target_name": settings.target_name,
        }
    except Exception as exc:
        logger.error("start_loop failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start loop: {exc}")


@router.post("/stop")
async def stop_loop() -> Dict[str, str]:
    """Signal the continuous loop to stop gracefully.

    Returns:
        Confirmation message with cycle count at stop time.
    """
    sched = _get_scheduler()

    if not sched.running:
        return {"message": "Loop is not running", "status": "not_running"}

    try:
        sched.stop()
        cycle_count = sched.cycle_count

        # Cancel the background task if it's still around
        global _continuous_task
        if _continuous_task is not None and not _continuous_task.done():
            _continuous_task.cancel()
            try:
                await _continuous_task
            except asyncio.CancelledError:
                pass
            _continuous_task = None

        logger.info("Continuous loop stopped after %d cycles", cycle_count)
        return {
            "message": "Loop stop signal sent",
            "cycle_count": cycle_count,
        }
    except Exception as exc:
        logger.error("stop_loop failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to stop loop: {exc}")


@router.post("/step")
async def step_loop() -> Dict[str, Any]:
    """Run one manual optimization cycle.

    Returns:
        The cycle record produced by the single cycle.
    """
    sched = _get_scheduler()

    try:
        record = await sched.run_single_cycle()
        if record is None:
            return {
                "success": False,
                "cycle_number": sched.cycle_count,
                "record": None,
                "error": "Cycle returned None (likely an error occurred)",
            }

        # Convert dataclass to dict for JSON serialization
        record_dict = {
            "cycle_id": record.cycle_id,
            "timestamp": record.timestamp.isoformat() if record.timestamp else None,
            "parent_smiles": record.parent_smiles,
            "proposed_modification": record.proposed_modification,
            "new_smiles": record.new_smiles,
            "new_fp": record.new_fp,
            "predicted_affinity_nm": record.predicted_affinity_nm,
            "improvement": record.improvement,
            "is_best_so_far": record.is_best_so_far,
            "lesson": record.lesson,
            "fingerprint_diff": record.fingerprint_diff,
        }
        return {
            "success": True,
            "cycle_number": record.cycle_id,
            "record": record_dict,
            "error": None,
        }
    except Exception as exc:
        logger.error("step_loop failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cycle failed: {exc}")


@router.get("/cognition")
async def get_cognition() -> Dict[str, Any]:
    """Return the full cognition store content.

    Returns:
        Complete dictionary representation of the cognition store.
    """
    sched = _get_scheduler()
    try:
        return sched.cognition_store.to_dict()
    except Exception as exc:
        logger.error("get_cognition failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get cognition store: {exc}"
        )


@router.get("/cognition/summary")
async def get_cognition_summary() -> Dict[str, str]:
    """Return a human-readable summary of accumulated lessons.

    Returns:
        Dictionary with a ``summary`` text field.
    """
    sched = _get_scheduler()
    try:
        summary = sched.cognition_store.get_lessons_summary(n_recent=10)
        return {"summary": summary}
    except Exception as exc:
        logger.error("get_cognition_summary failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get lessons summary: {exc}"
        )


# ---------------------------------------------------------------------------
# Background task wrapper
# ---------------------------------------------------------------------------


async def _run_continuous_wrapper() -> None:
    """Wrap ``run_continuous`` so exceptions don't go unhandled."""
    sched = _get_scheduler()
    try:
        await sched.run_continuous()
    except asyncio.CancelledError:
        logger.info("Continuous loop task cancelled")
    except Exception as exc:
        logger.error("Continuous loop crashed: %s", exc, exc_info=True)
