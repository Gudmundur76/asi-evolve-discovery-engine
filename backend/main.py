"""ASI-Evolve Discovery Engine — FastAPI application factory.

This module creates and configures the FastAPI application with:
- CORS middleware for cross-origin requests
- API router registration for all endpoint groups
- Database table creation on startup
- Health check with model file verification
- Root redirect to API documentation
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.api import (
    candidates_router,
    discovery_router,
    evidence_router,
    loop_router,
)
from backend.config import settings
from backend.database.session import create_tables, dispose_engine
from backend.core.auto_bootstrap import ensure_model_ready

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler.

    On startup:
        - Ensures database tables exist.
        - Logs configuration summary.

    On shutdown:
        - Gracefully disposes the database engine.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("ASI-Evolve Discovery Engine starting up")
    logger.info("=" * 60)
    logger.info("Target: %s (%s)", settings.target_name, settings.target_chembl_id)
    logger.info("Fingerprint: radius=%d, nbits=%d", settings.fingerprint_radius, settings.fingerprint_nbits)
    logger.info("Data directory: %s", settings.data_dir)
    logger.info("Model path: %s", settings.model_path)

    await create_tables()

    # Auto-bootstrap: train model from ChEMBL if model.pkl is missing.
    # No-op on subsequent starts when the model file already exists.
    await ensure_model_ready()

    yield
    # Shutdown
    await dispose_engine()
    logger.info("ASI-Evolve Discovery Engine shut down")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ASI-Evolve Discovery Engine",
    version="1.0.0",
    description=(
        "AI-driven molecular drug discovery with three-agent optimization loop. "
        "Optimises molecular candidates for target binding affinity using "
        "a Researcher-Engineer-Analyzer agent pipeline."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------

app.include_router(discovery_router, prefix="/api/discoveries")
app.include_router(loop_router, prefix="/api/loop")
app.include_router(candidates_router, prefix="/api/candidates")
app.include_router(evidence_router, prefix="/api/evidence")

# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint.

    Verifies that critical model files exist on disk.

    Returns:
        Dictionary with ``status`` and ``model_ready`` flag.
    """
    model_exists = Path(settings.model_path).exists()
    metadata_exists = Path(settings.model_metadata_path).exists()

    status = "healthy" if model_exists else "degraded"
    model_ready = model_exists and metadata_exists

    return {
        "status": status,
        "model_ready": model_ready,
        "model_path": str(settings.model_path),
        "model_exists": model_exists,
        "metadata_exists": metadata_exists,
        "target_chembl_id": settings.target_chembl_id,
        "target_name": settings.target_name,
        "version": "1.0.0",
    }


@app.get("/")
async def root() -> RedirectResponse:
    """Redirect root to the interactive API documentation.

    Returns:
        307 redirect to ``/docs``.
    """
    return RedirectResponse(url="/docs", status_code=307)
