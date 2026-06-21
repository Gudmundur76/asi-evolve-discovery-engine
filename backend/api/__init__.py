"""API router package.

Exports all FastAPI routers for inclusion in the main application.
"""

from backend.api.discovery import router as discovery_router
from backend.api.loop_status import router as loop_router
from backend.api.candidates import router as candidates_router
from backend.api.evidence import router as evidence_router

__all__ = [
    "discovery_router",
    "loop_router",
    "candidates_router",
    "evidence_router",
]
