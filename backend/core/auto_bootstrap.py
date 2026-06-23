"""Auto-bootstrap: ensure the affinity model exists before the first request.

On a fresh Manus deployment the ``data/`` directory is empty — no model has
been trained.  This module provides :func:`ensure_model_ready`, called from
the FastAPI lifespan handler, which:

1. Checks whether ``settings.model_path`` already exists.
2. If it does → no-op (fast path, < 1 ms).
3. If it does not → runs the full training pipeline in a thread-pool executor
   so the asyncio event loop stays responsive.

The training pipeline:
    a. Fetches HIV-1 protease bioactivities from ChEMBL.
    b. Encodes SMILES as Morgan fingerprints.
    c. Trains a RandomForest regressor (log10 affinity).
    d. Saves ``model.pkl`` + ``model_metadata.json`` to ``settings.data_dir``.

If training fails (e.g. ChEMBL is unreachable), a warning is logged and the
application starts anyway — the ``/health`` endpoint will report
``model_ready: false`` and the loop endpoints will return 503 with a clear
message instead of a cryptic 500.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import numpy as np

from backend.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def ensure_model_ready() -> bool:
    """Ensure the model file exists, training from ChEMBL if necessary.

    Returns:
        ``True`` if the model is ready (existed or was just trained),
        ``False`` if training failed and the model is still absent.
    """
    model_path = Path(settings.model_path)

    if model_path.exists():
        logger.info("Model already exists at %s — skipping auto-bootstrap.", model_path)
        return True

    logger.info(
        "Model not found at %s — starting auto-bootstrap training from ChEMBL …",
        model_path,
    )

    loop = asyncio.get_event_loop()
    try:
        success = await loop.run_in_executor(None, _train_blocking)
    except Exception as exc:
        logger.warning(
            "Auto-bootstrap training raised an unexpected error: %s — "
            "application will start without a trained model.",
            exc,
            exc_info=True,
        )
        return False

    if success:
        logger.info("Auto-bootstrap complete — model ready at %s", model_path)
    else:
        logger.warning(
            "Auto-bootstrap training failed — application will start without a "
            "trained model.  POST /api/loop/start will return 503 until the model "
            "is available.  You can trigger training manually via "
            "POST /api/loop/train."
        )
    return success


# ---------------------------------------------------------------------------
# Blocking training pipeline (runs in thread-pool executor)
# ---------------------------------------------------------------------------


def _train_blocking() -> bool:
    """Run the full training pipeline synchronously.

    Returns:
        ``True`` on success, ``False`` on any recoverable failure.
    """
    try:
        from backend.core.chembl_client import ChEMBLClient
        from backend.core.fingerprint import FingerprintEncoder
        from backend.core.model_trainer import ModelTrainer
    except ImportError as exc:
        logger.error("Cannot import training dependencies: %s", exc)
        return False

    # 1. Ensure data directory exists
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 2. Fetch ChEMBL activities
    client = ChEMBLClient(target_chembl_id=settings.target_chembl_id)
    try:
        activities = client.fetch_activities(
            activity_type=settings.activity_type,
            limit=settings.activity_limit,
        )
    except Exception as exc:
        logger.warning("ChEMBL fetch failed: %s", exc)
        return False

    if activities.empty:
        logger.warning("ChEMBL returned no activities — cannot train.")
        return False

    logger.info("ChEMBL returned %d activities for %s", len(activities), settings.target_chembl_id)

    # 3. Encode SMILES → fingerprints
    encoder = FingerprintEncoder(
        radius=settings.fingerprint_radius,
        n_bits=settings.fingerprint_nbits,
    )
    fingerprints: list = []
    affinities: list = []
    for _, row in activities.iterrows():
        fp = encoder.smiles_to_fp(row["canonical_smiles"])
        if fp is not None:
            fingerprints.append(fp)
            affinities.append(row["standard_value"])

    if len(fingerprints) < 10:
        logger.warning(
            "Only %d valid fingerprints — need at least 10 to train.", len(fingerprints)
        )
        return False

    X = np.vstack(fingerprints)
    y = np.log10(np.maximum(affinities, 1e-6))
    logger.info("Training on %d molecules (fingerprint shape: %s)", len(fingerprints), X.shape)

    # 4. Train
    trainer = ModelTrainer(
        model_type=settings.model_type,
        n_estimators=settings.n_estimators,
        max_depth=settings.max_depth,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )
    metrics = trainer.train(X, y)

    # 5. Save model + metadata
    trainer.save(settings.model_path)
    with open(settings.model_metadata_path, "w") as fh:
        json.dump(trainer.metadata, fh, indent=2)

    logger.info(
        "Auto-bootstrap training complete: R²=%.4f RMSE=%.4f n_train=%d",
        metrics["r2_score"],
        metrics["rmse"],
        metrics["n_train"],
    )
    return True
