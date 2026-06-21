"""Engineer agent for the ASI-Evolve molecular discovery engine.

The Engineer agent applies proposed molecular modifications to fingerprints.
It implements four modification strategies: bit_flip, guided_mutation,
crossover, and exploration. Each strategy manipulates the fingerprint bit
vector in a different way to generate a novel candidate molecule.
"""

import logging
import random
from typing import List, Tuple

import numpy as np

from backend.core import FingerprintEncoder

logger = logging.getLogger(__name__)


class EngineerAgent:
    """Agent that applies molecular modifications to fingerprints.

    Routes modification requests to the appropriate strategy implementation
    and tracks which bits were changed for downstream analysis.
    """

    def __init__(self, fingerprint_encoder: FingerprintEncoder) -> None:
        """Initialize the Engineer agent.

        Args:
            fingerprint_encoder: Encoder for SMILES <-> fingerprint conversions.
        """
        self.encoder = fingerprint_encoder
        logger.debug(
            "EngineerAgent initialized (fp_size=%d)", self.encoder.n_bits
        )

    def apply_modification(
        self, base_fp: np.ndarray, modification: dict
    ) -> Tuple[np.ndarray, List[int]]:
        """Apply a proposed modification to a base fingerprint.

        Args:
            base_fp: Dense binary fingerprint to modify.
            modification: Dict with keys: strategy, target_bits, rationale, confidence.

        Returns:
            Tuple of (new_fp, changed_bit_indices) where new_fp is the modified
            dense fingerprint and changed_bit_indices lists which positions differ.

        Raises:
            ValueError: If the strategy is not recognized.
        """
        strategy = modification.get("strategy", "exploration")
        target_bits = modification.get("target_bits", [])

        logger.info("Applying strategy: %s", strategy)

        if strategy == "bit_flip":
            new_fp = self._bit_flip(base_fp, target_bits)
        elif strategy == "guided_mutation":
            patterns = modification.get("patterns", {})
            n_bits = modification.get("n_bits", 3)
            new_fp = self._guided_mutation(base_fp, patterns, n_bits)
        elif strategy == "crossover":
            other_fp = modification.get("other_fp")
            if other_fp is None:
                # Attempt to construct from target_bits + base
                new_fp = self._uniform_crossover(base_fp, target_bits)
            else:
                new_fp = self._uniform_crossover(base_fp, other_fp)
        elif strategy == "exploration":
            n_bits = modification.get("n_bits", 5)
            new_fp = self._exploration_mutation(base_fp, n_bits)
        else:
            raise ValueError(f"Unknown modification strategy: {strategy}")

        # Determine which bits changed
        changed_bits = self._compute_diff(base_fp, new_fp)

        logger.info(
            "Modification complete: %d bits changed (strategy=%s)",
            len(changed_bits),
            strategy,
        )
        return new_fp, changed_bits

    # ------------------------------------------------------------------ #
    # Strategy implementations
    # ------------------------------------------------------------------ #

    def _bit_flip(self, fp: np.ndarray, target_bits: List[int]) -> np.ndarray:
        """Flip specified bits in the fingerprint.

        Args:
            fp: Dense binary fingerprint.
            target_bits: List of bit positions to flip (0-indexed).

        Returns:
            New fingerprint with specified bits toggled.
        """
        new_fp = fp.copy()
        valid_bits = [b for b in target_bits if 0 <= b < len(new_fp)]
        if not valid_bits:
            logger.warning("No valid bits to flip, returning unchanged fingerprint")
            return new_fp
        new_fp[valid_bits] = 1 - new_fp[valid_bits]
        return new_fp

    def _guided_mutation(
        self, fp: np.ndarray, patterns: dict, n_bits: int = 3
    ) -> np.ndarray:
        """Probabilistically select bits weighted by historical improvement and flip them.

        Bits with more negative avg_improvement (larger affinity reduction) get
        higher weight for being flipped.

        Args:
            fp: Dense binary fingerprint.
            patterns: Per-bit statistics dict with avg_improvement values.
            n_bits: Number of bits to mutate.

        Returns:
            New fingerprint with probabilistically selected bits flipped.
        """
        new_fp = fp.copy()

        if not patterns:
            logger.warning("No patterns for guided mutation, falling back to random")
            return self._exploration_mutation(new_fp, n_bits)

        # Build weighted list of all bits
        all_bits = list(range(len(new_fp)))
        weights = np.ones(len(all_bits), dtype=float) * 0.01  # Small baseline

        for bit_key, stats in patterns.items():
            try:
                bit_pos = int(bit_key)
            except (ValueError, TypeError):
                bit_pos = bit_key if isinstance(bit_key, int) else None
            if bit_pos is None or not (0 <= bit_pos < len(new_fp)):
                continue

            # More negative avg_improvement = better = higher weight
            avg_imp = stats.get("avg_improvement", 0.0)
            # Weight inversely proportional to avg_improvement
            # (negative improvement = we want to flip this bit)
            weight = max(0.01, -avg_imp + 0.1)
            weights[bit_pos] = weight

        # Normalize
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(all_bits)) / len(all_bits)

        # Sample without replacement
        n_flip = min(n_bits, len(new_fp))
        try:
            selected = np.random.choice(all_bits, size=n_flip, replace=False, p=weights)
        except ValueError:
            selected = random.sample(all_bits, n_flip)

        new_fp[selected] = 1 - new_fp[selected]
        return new_fp

    def _uniform_crossover(
        self, fp: np.ndarray, other: np.ndarray
    ) -> np.ndarray:
        """Randomly swap segments between two fingerprints.

        Args:
            fp: First dense binary fingerprint (current best).
            other: Second dense binary fingerprint (other candidate).

        Returns:
            New fingerprint combining segments from both parents.
        """
        if len(fp) != len(other):
            raise ValueError(
                f"Fingerprint length mismatch: {len(fp)} vs {len(other)}"
            )

        # Create crossover mask: random segments
        n_segments = random.randint(2, 5)
        segment_size = len(fp) // n_segments
        new_fp = fp.copy()

        for seg in range(n_segments):
            if random.random() < 0.5:
                start = seg * segment_size
                end = start + segment_size if seg < n_segments - 1 else len(fp)
                new_fp[start:end] = other[start:end]

        return new_fp

    def _exploration_mutation(self, fp: np.ndarray, n_bits: int = 5) -> np.ndarray:
        """Purely random bit flips for exploration.

        Args:
            fp: Dense binary fingerprint.
            n_bits: Number of bits to randomly flip.

        Returns:
            New fingerprint with random bits toggled.
        """
        new_fp = fp.copy()
        n_flip = min(n_bits, len(new_fp))
        selected = random.sample(range(len(new_fp)), n_flip)
        new_fp[selected] = 1 - new_fp[selected]
        return new_fp

    # ------------------------------------------------------------------ #
    # Utility methods
    # ------------------------------------------------------------------ #

    def fp_to_sparse(self, fp: np.ndarray) -> List[int]:
        """Convert a dense fingerprint to sparse representation.

        Args:
            fp: Dense binary fingerprint.

        Returns:
            List of integer positions where the fingerprint is 1.
        """
        return self.encoder.fp_to_sparse(fp)

    @staticmethod
    def _compute_diff(fp1: np.ndarray, fp2: np.ndarray) -> List[int]:
        """Compute the indices of differing bits between two fingerprints.

        Args:
            fp1: Original fingerprint.
            fp2: Modified fingerprint.

        Returns:
            Sorted list of bit positions that differ.
        """
        diff = np.where(fp1 != fp2)[0]
        return diff.tolist()
