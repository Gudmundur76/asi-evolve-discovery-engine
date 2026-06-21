"""High-level affinity predictor that combines fingerprint encoding
with a trained ML model.

Supports single-item, batch, and SMILES-based predictions while
clamping outputs to physically reasonable ranges.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from backend.config import Settings, settings
from backend.core.fingerprint import FingerprintEncoder

logger = logging.getLogger(__name__)


class PredictorError(Exception):
    """Raised for prediction-time errors."""

    pass


class AffinityPredictor:
    """Predict binding affinity (nM) from molecular fingerprints or SMILES.

    Loads a trained model and its training metadata at construction time.

    Args:
        model_path: Path to the pickled model file.  Uses
            ``settings.model_path`` when *None*.
        encoder: Shared :class:`FingerprintEncoder` instance.  A new one
            is created when *None*.
        settings_override: Optional settings object for testing.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        encoder: Optional[FingerprintEncoder] = None,
        settings_override: Optional[Settings] = None,
    ) -> None:
        self._cfg = settings_override or settings
        self.model_path = Path(model_path or self._cfg.model_path)
        self.metadata_path = self._cfg.model_metadata_path
        self.encoder = encoder or FingerprintEncoder(
            radius=self._cfg.fingerprint_radius,
            n_bits=self._cfg.fingerprint_nbits,
        )

        self._model: Any = None
        self._metadata: Dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        """Load model and metadata from disk."""
        if not self.model_path.exists():
            raise PredictorError(
                f"Model file not found: {self.model_path}. "
                "Run scripts/train_model.py first."
            )

        with open(self.model_path, "rb") as fh:
            self._model = pickle.load(fh)

        if self.metadata_path.exists():
            with open(self.metadata_path) as fh:
                self._metadata = json.load(fh)
        else:
            logger.warning("No metadata file found at %s", self.metadata_path)
            self._metadata = {
                "best_affinity_nM": 0.001,
                "n_train": 0,
                "model_type": "unknown",
            }

        logger.info(
            "Predictor ready (model=%s, n_train=%d)",
            self._metadata.get("model_type", "unknown"),
            self._metadata.get("n_train", 0),
        )

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #
    @property
    def best_training_affinity(self) -> float:
        """Best (lowest) affinity observed during training, in nM."""
        return float(self._metadata.get("best_affinity_nM", 0.001))

    def is_better_than_training(self, affinity_nm: float) -> bool:
        """Return *True* if *affinity_nm* is lower (better) than the best
        value seen during training.
        """
        return affinity_nm < self.best_training_affinity

    # ------------------------------------------------------------------ #
    #  Prediction API
    # ------------------------------------------------------------------ #
    def predict(self, fp: np.ndarray) -> float:
        """Predict affinity in nM from a single fingerprint vector.

        The model outputs log10(nM); this method converts back to nM
        and clamps to a minimum of 0.001 nM.

        Args:
            fp: Dense fingerprint array of shape ``(n_bits,)``.

        Returns:
            Predicted affinity in nM.
        """
        if self._model is None:
            raise PredictorError("Model not loaded.")

        # Ensure 2-D input for sklearn
        if fp.ndim == 1:
            fp = fp.reshape(1, -1)

        log_pred = self._model.predict(fp)[0]
        affinity = 10.0 ** log_pred
        return max(affinity, 0.001)

    def predict_batch(self, fps: np.ndarray) -> List[float]:
        """Predict affinities for a batch of fingerprints.

        Args:
            fps: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            List of predicted affinities in nM.
        """
        if fps.ndim == 1:
            return [self.predict(fps)]
        log_preds = self._model.predict(fps)
        affinities = [max(10.0 ** lp, 0.001) for lp in log_preds]
        return affinities

    def predict_smiles(self, smiles: str) -> Optional[float]:
        """End-to-end prediction from a SMILES string.

        Args:
            smiles: Canonical SMILES.

        Returns:
            Predicted affinity in nM, or *None* if SMILES parsing fails.
        """
        fp = self.encoder.smiles_to_fp(smiles)
        if fp is None:
            logger.warning("Failed to encode SMILES: %s", smiles)
            return None
        return self.predict(fp)
