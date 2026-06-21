#!/usr/bin/env python3
"""One-shot training script for the molecular discovery engine.

Usage::

    python scripts/train_model.py

Steps:
    1. Load configuration.
    2. Fetch bioactivities from ChEMBL.
    3. Convert SMILES to Morgan fingerprints.
    4. Train a RandomForest regressor (log10 affinity).
    5. Save model + metadata and print evaluation metrics.
"""

import logging
import sys

import numpy as np

# Allow imports from the repo root
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.core.chembl_client import ChEMBLClient
from backend.core.fingerprint import FingerprintEncoder
from backend.core.model_trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_model")


def main() -> int:
    """Run the full training pipeline."""
    logger.info("=" * 60)
    logger.info(" Molecular Discovery Engine — Model Training Pipeline ")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Configuration
    # ------------------------------------------------------------------
    logger.info("Target: %s (%s)", settings.target_chembl_id, settings.target_name)
    logger.info("Activity type: %s", settings.activity_type)
    logger.info("Fingerprint: Morgan r=%d, %d bits", settings.fingerprint_radius, settings.fingerprint_nbits)

    # ------------------------------------------------------------------
    # 2. Fetch ChEMBL activities
    # ------------------------------------------------------------------
    client = ChEMBLClient(target_chembl_id=settings.target_chembl_id)

    try:
        target_info = client.fetch_target_info()
        logger.info(
            "Target info: %s | %s | %s",
            target_info["target_name"],
            target_info["organism"],
            target_info["target_type"],
        )
    except Exception as exc:
        logger.warning("Could not fetch target info: %s", exc)

    activities = client.fetch_activities(
        activity_type=settings.activity_type,
        limit=settings.activity_limit,
    )

    if activities.empty:
        logger.error("No activities retrieved — cannot train.")
        return 1

    logger.info("Retrieved %d unique activities", len(activities))

    # ------------------------------------------------------------------
    # 3. Encode SMILES → fingerprints
    # ------------------------------------------------------------------
    encoder = FingerprintEncoder(
        radius=settings.fingerprint_radius,
        n_bits=settings.fingerprint_nbits,
    )

    fingerprints: list = []
    affinities: list = []

    for _, row in activities.iterrows():
        smiles = row["canonical_smiles"]
        fp = encoder.smiles_to_fp(smiles)
        if fp is not None:
            fingerprints.append(fp)
            affinities.append(row["standard_value"])

    if len(fingerprints) < 10:
        logger.error(
            "Only %d valid fingerprints — need at least 10 to train.", len(fingerprints)
        )
        return 1

    X = np.vstack(fingerprints)
    # log10 transform: model predicts log10(nM)
    y = np.log10(np.maximum(affinities, 1e-6))

    logger.info("Feature matrix: %s | Labels: %s", X.shape, y.shape)
    logger.info("Affinity range: %.2f – %.2f log10(nM)", y.min(), y.max())

    # ------------------------------------------------------------------
    # 4. Train model
    # ------------------------------------------------------------------
    trainer = ModelTrainer(
        model_type=settings.model_type,
        n_estimators=settings.n_estimators,
        max_depth=settings.max_depth,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )

    metrics = trainer.train(X, y)

    # ------------------------------------------------------------------
    # 5. Save model + metadata
    # ------------------------------------------------------------------
    trainer.save(settings.model_path)

    # Also save metadata to the configured metadata path
    import json
    with open(settings.model_metadata_path, "w") as fh:
        json.dump(trainer.metadata, fh, indent=2)

    # ------------------------------------------------------------------
    # 6. Print evaluation metrics
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("EVALUATION METRICS")
    logger.info("-" * 40)
    logger.info("R2  score : %.4f", metrics["r2_score"])
    logger.info("RMSE      : %.4f", metrics["rmse"])
    logger.info("MAE       : %.4f", metrics["mae"])
    logger.info("Train set : %d molecules", metrics["n_train"])
    logger.info("Test set  : %d molecules", metrics["n_test"])
    logger.info("Model file: %s", metrics["model_path"])
    logger.info("Metadata  : %s", settings.model_metadata_path)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
