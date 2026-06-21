"""Mock affinity predictor for testing the agent loop.

In production, this would use a trained GNN or similar ML model to predict
binding affinity (pKi/pKd) from molecular fingerprints. This mock version
uses a deterministic function based on fingerprint features.
"""

import numpy as np
from typing import List, Optional, Union


class AffinityPredictor:
    """Predicts binding affinity from molecular fingerprints.

    Production implementation would use a trained neural network.
    This mock version uses a deterministic heuristic based on fingerprint
    bit counts in specific regions to simulate structure-activity relationships.
    """

    def __init__(self, model_path: Optional[str] = None, fp_size: int = 2048) -> None:
        """Initialize the affinity predictor.

        Args:
            model_path: Path to a saved model (ignored in mock).
            fp_size: Expected fingerprint size.
        """
        self.model_path = model_path
        self.fp_size = fp_size

    def predict(self, fp: np.ndarray) -> float:
        """Predict binding affinity (in nM) for a single fingerprint.

        Lower values = better binding. The mock heuristic simulates SAR by
        rewarding certain bit patterns and penalizing others.

        Args:
            fp: Binary fingerprint array.

        Returns:
            Predicted affinity in nanomolar (nM).
        """
        if fp is None or len(fp) == 0:
            return 10000.0  # Very poor affinity for empty fp

        fp_arr = np.asarray(fp).astype(np.uint8)
        n_bits = int(fp_arr.sum())

        if n_bits == 0:
            return 10000.0

        # Base affinity: more bits = generally better (more features)
        # but with diminishing returns and noise
        base = 1000.0 - 20.0 * min(n_bits, 40)  # Reward up to ~40 bits

        # Region-based bonuses (simulating pharmacophore features)
        # Hydrogen bond donors (bits 0-200)
        hbd_bits = int(fp_arr[0:200].sum())
        hbd_bonus = -15.0 * min(hbd_bits, 5)  # Up to 5 HBDs help

        # Hydrophobic region (bits 200-600)
        hydro_bits = int(fp_arr[200:600].sum())
        hydro_bonus = -10.0 * min(hydro_bits, 15)  # Hydrophobic contacts help

        # Aromatic interactions (bits 600-900)
        arom_bits = int(fp_arr[600:900].sum())
        arom_bonus = -12.0 * min(arom_bits, 8)  # Pi-stacking helps

        # Electrostatic (bits 900-1100)
        elec_bits = int(fp_arr[900:1100].sum())
        elec_bonus = -8.0 * min(elec_bits, 6)

        # Penalty for too many bits (overfitting / too large)
        size_penalty = 5.0 * max(0, n_bits - 50)

        # Noise based on fingerprint content for deterministic but non-trivial behavior
        noise_seed = int(np.dot(fp_arr[:100], np.arange(1, 101)))
        np.random.seed(noise_seed)
        noise = np.random.normal(0, 50.0)

        affinity = (
            base + hbd_bonus + hydro_bonus + arom_bonus + elec_bonus
            + size_penalty + noise
        )

        # Clamp to reasonable range [0.1, 10000] nM
        return float(np.clip(affinity, 0.1, 10000.0))

    def predict_batch(self, fps: np.ndarray) -> List[float]:
        """Predict affinity for a batch of fingerprints.

        Args:
            fps: 2D array of fingerprints, shape (n_molecules, fp_size).

        Returns:
            List of predicted affinities in nM.
        """
        return [self.predict(fp) for fp in fps]
