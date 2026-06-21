"""Mock fingerprint encoder for testing the agent loop.

In production, this would use RDKit or similar to generate Morgan fingerprints
from SMILES strings. For the agent loop demonstration, we use a simple
hash-based fingerprint generator that produces consistent bit vectors.
"""

import hashlib
import numpy as np
from typing import List


class FingerprintEncoder:
    """Encodes SMILES strings into fixed-length binary fingerprints.

    Production implementation would use RDKit Morgan fingerprints.
    This mock version uses deterministic hashing for reproducibility.
    """

    def __init__(self, fp_size: int = 2048, radius: int = 2) -> None:
        """Initialize the fingerprint encoder.

        Args:
            fp_size: Length of the binary fingerprint vector.
            radius: Morgan fingerprint radius (mocked, kept for API compat).
        """
        self.fp_size = fp_size
        self.radius = radius

    def encode(self, smiles: str) -> np.ndarray:
        """Encode a SMILES string into a binary fingerprint.

        Args:
            smiles: SMILES representation of a molecule.

        Returns:
            Binary numpy array of shape (fp_size,) with dtype uint8.
        """
        # Deterministic seed from SMILES
        seed = int(hashlib.md5(smiles.encode()).hexdigest(), 16)
        rng = np.random.default_rng(seed)
        fp = rng.integers(0, 2, size=self.fp_size, dtype=np.uint8)
        # Ensure at least some bits are set
        if fp.sum() == 0:
            fp[seed % self.fp_size] = 1
        return fp

    def smiles_to_fp(self, smiles: str) -> np.ndarray:
        """Alias for encode()."""
        return self.encode(smiles)

    def encode_batch(self, smiles_list: List[str]) -> np.ndarray:
        """Encode a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            2D numpy array of shape (n_molecules, fp_size).
        """
        return np.array([self.encode(s) for s in smiles_list])

    def sparse_to_dense(self, sparse_fp: List[int]) -> np.ndarray:
        """Convert a sparse fingerprint (list of set bit positions) to dense vector.

        Args:
            sparse_fp: List of integer bit positions that are set to 1.

        Returns:
            Dense binary numpy array of shape (fp_size,).
        """
        dense = np.zeros(self.fp_size, dtype=np.uint8)
        if sparse_fp:
            # Filter valid indices
            valid = [b for b in sparse_fp if 0 <= b < self.fp_size]
            dense[valid] = 1
        return dense

    @staticmethod
    def dense_to_sparse(dense_fp: np.ndarray) -> List[int]:
        """Convert a dense binary fingerprint to sparse representation.

        Args:
            dense_fp: Dense binary numpy array.

        Returns:
            List of integer positions where the fingerprint is 1.
        """
        return np.where(dense_fp)[0].tolist()

    def fingerprint_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Compute Tanimoto similarity between two fingerprints.

        Args:
            fp1: First fingerprint.
            fp2: Second fingerprint.

        Returns:
            Tanimoto similarity coefficient in [0, 1].
        """
        fp1_b = fp1.astype(bool)
        fp2_b = fp2.astype(bool)
        intersection = np.logical_and(fp1_b, fp2_b).sum()
        union = np.logical_or(fp1_b, fp2_b).sum()
        if union == 0:
            return 0.0
        return float(intersection) / float(union)
