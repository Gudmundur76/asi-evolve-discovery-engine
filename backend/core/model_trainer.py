"""Scikit-learn model training wrapper with evaluation and persistence.

Trains a RandomForestRegressor on fingerprint vectors to predict
log10-transformed bioactivity values.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend.config import settings

logger = logging.getLogger(__name__)


class ModelTrainerError(Exception):
    """Base exception for model-trainer errors."""

    pass


class ModelTrainer:
    """Train and evaluate a Random-Forest regressor on molecular fingerprints.

    The caller is responsible for log10-transforming the affinity labels
    before calling :meth:`train`.

    Args:
        model_type: Identifier for the model family (default "random_forest").
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of each tree.
        test_size: Fraction held out for evaluation.
        random_state: Reproducibility seed.
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        n_estimators: int = 200,
        max_depth: int = 20,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.test_size = test_size
        self.random_state = random_state
        self.model: Optional[RandomForestRegressor] = None
        self.metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Train the model and evaluate on a held-out test set.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target vector — **must already be log10(affinity_in_nM)**.

        Returns:
            Dictionary with keys:
            ``model_path, r2_score, rmse, mae, n_train, n_test``.
        """
        if X.shape[0] != y.shape[0]:
            raise ModelTrainerError(
                f"X and y have mismatched lengths: {X.shape[0]} vs {y.shape[0]}"
            )
        if X.shape[0] < 10:
            raise ModelTrainerError(
                f"Too few samples to train: {X.shape[0]} (need >= 10)"
            )

        logger.info(
            "Training %s (n=%d, features=%d) ...",
            self.model_type,
            X.shape[0],
            X.shape[1],
        )

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        # Fit model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        train_metrics = self.evaluate(X_train, y_train)
        test_metrics = self.evaluate(X_test, y_test)

        self.metadata = {
            "model_type": self.model_type,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X.shape[1]),
            "train_r2": train_metrics["r2"],
            "test_r2": test_metrics["r2"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "best_affinity_nM": float(10 ** y_train.min()),
            "worst_affinity_nM": float(10 ** y_train.max()),
            "mean_affinity_nM": float(10 ** y_train.mean()),
        }

        logger.info(
            "Training complete — R2=%.4f (train) / %.4f (test), RMSE=%.4f",
            train_metrics["r2"],
            test_metrics["r2"],
            test_metrics["rmse"],
        )

        return {
            "model_path": str(settings.model_path),
            "r2_score": test_metrics["r2"],
            "rmse": test_metrics["rmse"],
            "mae": test_metrics["mae"],
            "n_train": self.metadata["n_train"],
            "n_test": self.metadata["n_test"],
        }

    # ------------------------------------------------------------------ #
    #  Evaluation
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate the trained model on the given data.

        Args:
            X: Feature matrix.
            y: Ground-truth labels (log10 scale).

        Returns:
            Dictionary with ``r2``, ``rmse``, ``mae``, and ``predictions``.
        """
        if self.model is None:
            raise ModelTrainerError("No trained model available. Call train() first.")

        y_pred = self.model.predict(X)

        return {
            "r2": float(r2_score(y, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "mae": float(mean_absolute_error(y, y_pred)),
            "predictions": y_pred,
        }

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Union[str, Path]) -> None:
        """Pickle the trained model and write JSON metadata alongside it.

        Args:
            path: Destination file path (``.pkl``).
        """
        if self.model is None:
            raise ModelTrainerError("No trained model to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        with open(path, "wb") as fh:
            pickle.dump(self.model, fh)

        # Save metadata
        meta_path = path.with_suffix(".json")
        if str(meta_path) == str(path):
            meta_path = path.with_name(path.stem + "_metadata.json")

        with open(meta_path, "w") as fh:
            json.dump(self.metadata, fh, indent=2)

        logger.info("Model saved to %s, metadata to %s", path, meta_path)

    def load_model(self, path: Union[str, Path]) -> None:
        """Unpickle a previously saved model.

        Args:
            path: Path to the ``.pkl`` file.
        """
        path = Path(path)
        if not path.exists():
            raise ModelTrainerError(f"Model file not found: {path}")

        with open(path, "rb") as fh:
            self.model = pickle.load(fh)

        logger.info("Model loaded from %s", path)
