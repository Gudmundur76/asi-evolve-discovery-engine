"""Analyzer agent for the ASI-Evolve molecular discovery engine.

The Analyzer agent evaluates newly generated molecular candidates by predicting
their binding affinity, comparing against parent molecules, distilling lessons
from the results, and updating the cognition store's statistical patterns.
"""

import logging
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from backend.agents.cognition_store import CognitionStore, CycleRecord
from backend.core import AffinityPredictor, FingerprintEncoder

logger = logging.getLogger(__name__)


class AnalyzerAgent:
    """Agent that evaluates molecular candidates and updates the cognition store.

    Pipeline:
    1. Predict affinity of the new fingerprint.
    2. Retrieve or predict the parent's affinity.
    3. Calculate improvement (negative = better binding).
    4. Check if this is the best affinity ever achieved.
    5. Distill a human-readable lesson.
    6. Update per-bit statistics in the cognition store.
    7. Record the cycle in the cognition store.
    """

    def __init__(
        self,
        affinity_predictor: AffinityPredictor,
        cognition_store: CognitionStore,
        fingerprint_encoder: FingerprintEncoder,
    ) -> None:
        """Initialize the Analyzer agent.

        Args:
            affinity_predictor: Model for predicting binding affinity.
            cognition_store: Shared knowledge store for recording cycles.
            fingerprint_encoder: Encoder for fingerprint conversions.
        """
        self.predictor = affinity_predictor
        self.store = cognition_store
        self.encoder = fingerprint_encoder
        logger.debug("AnalyzerAgent initialized")

    def analyze_candidate(
        self,
        new_fp: np.ndarray,
        parent_smiles: str,
        proposed_modification: dict,
        cycle_number: int,
        parent_fp: Optional[np.ndarray] = None,
    ) -> CycleRecord:
        """Analyze a candidate fingerprint and record the cycle.

        Args:
            new_fp: Dense binary fingerprint of the candidate molecule.
            parent_smiles: SMILES of the parent molecule.
            proposed_modification: Dict from the Researcher agent describing
                the strategy, target_bits, rationale, and confidence.
            cycle_number: Current optimization cycle number.
            parent_fp: Optional dense fingerprint of the parent. If None, the
                parent fingerprint is looked up from the store or re-encoded
                from parent_smiles.

        Returns:
            The completed CycleRecord added to the cognition store.
        """
        # Step 1: Predict affinity of new fingerprint
        new_affinity = self.predictor.predict(new_fp)
        logger.info("Predicted affinity for candidate: %.3f nM", new_affinity)

        # Step 2: Get parent's affinity and fingerprint
        parent_affinity, actual_parent_fp = self._resolve_parent(
            parent_smiles, parent_fp
        )
        logger.info("Parent affinity: %.3f nM", parent_affinity)

        # Step 3: Calculate improvement (negative = better)
        improvement = new_affinity - parent_affinity

        # Step 4: Check if best ever
        is_best = bool(new_affinity < self.store.best_affinity_ever)
        if is_best:
            logger.info(
                "*** NEW BEST AFFINITY: %.3f nM (improvement: %+.3f) ***",
                new_affinity,
                improvement,
            )

        # Step 5: Compute fingerprint diff using ACTUAL parent fingerprint
        fp_diff = self._compute_fingerprint_diff(actual_parent_fp, new_fp)

        # Step 6: Distill lesson
        lesson = self._distill_lesson(
            proposed_modification,
            parent_affinity,
            new_affinity,
            fp_diff,
            new_fp,
        )

        # Step 7: Update statistics using actual parent fingerprint
        changed_bits = self._get_changed_bits(actual_parent_fp, new_fp)
        self._update_statistics(self.store, changed_bits, improvement, new_fp)

        # Step 8: Generate a new SMILES (mock: perturb parent)
        new_smiles = self._generate_new_smiles(parent_smiles, cycle_number)

        # Create and record the cycle
        record = CycleRecord(
            cycle_id=cycle_number,
            timestamp=datetime.now(),
            parent_smiles=parent_smiles,
            proposed_modification=proposed_modification,
            new_smiles=new_smiles,
            new_fp=self.encoder.fp_to_sparse(new_fp),
            predicted_affinity_nm=new_affinity,
            improvement=improvement,
            is_best_so_far=is_best,
            lesson=lesson,
            fingerprint_diff=fp_diff,
        )

        self.store.add_cycle(record)
        logger.info(
            "Cycle %d recorded: affinity=%.3f nM, improvement=%+.3f, best=%s",
            cycle_number,
            new_affinity,
            improvement,
            is_best,
        )
        return record

    def _resolve_parent(
        self, parent_smiles: str, parent_fp: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """Resolve the parent's affinity and fingerprint.

        Priority:
        1. Use provided parent_fp if available.
        2. Look up in existing cycle records by matching parent_smiles to new_smiles.
        3. Fall back to encoding parent_smiles directly.

        Args:
            parent_smiles: SMILES of the parent molecule.
            parent_fp: Optional pre-computed parent fingerprint.

        Returns:
            Tuple of (parent_affinity, parent_fp_dense).
        """
        # Priority 1: caller provided the actual parent fingerprint
        if parent_fp is not None:
            affinity = self.predictor.predict(parent_fp)
            return affinity, parent_fp

        # Priority 2: look up in cycle records
        for cycle in reversed(self.store.cycles):
            if cycle.new_smiles == parent_smiles:
                # Reconstruct the fingerprint from the stored sparse representation
                fp_dense = np.zeros(self.encoder.n_bits, dtype=np.float32)
                fp_dense[list(cycle.new_fp)] = 1.0
                return cycle.predicted_affinity_nm, fp_dense

        # Priority 3: encode from SMILES
        fp_dense = self.encoder.smiles_to_fp(parent_smiles)
        affinity = self.predictor.predict(fp_dense)
        return affinity, fp_dense

    def _compute_fingerprint_diff(
        self, parent_fp: np.ndarray, new_fp: np.ndarray
    ) -> List[int]:
        """Find bit positions that differ between parent and candidate.

        Args:
            parent_fp: Parent fingerprint.
            new_fp: Candidate fingerprint.

        Returns:
            List of bit positions that changed.
        """
        diff = np.where(parent_fp != new_fp)[0]
        return diff.tolist()

    def _get_changed_bits(
        self, parent_fp: np.ndarray, new_fp: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Get list of (bit_position, new_value) for changed bits.

        Args:
            parent_fp: Parent fingerprint.
            new_fp: Candidate fingerprint.

        Returns:
            List of (bit_position, new_value) tuples.
        """
        changed = []
        for i in range(len(parent_fp)):
            if parent_fp[i] != new_fp[i]:
                changed.append((i, int(new_fp[i])))
        return changed

    def _distill_lesson(
        self,
        modification: dict,
        old_affinity: float,
        new_affinity: float,
        changed_bits: List[int],
        new_fp: np.ndarray,
    ) -> str:
        """Generate a human-readable lesson from a cycle's results.

        Args:
            modification: Proposed modification dict.
            old_affinity: Parent molecule's predicted affinity (nM).
            new_affinity: Candidate's predicted affinity (nM).
            changed_bits: Bit positions that were modified.
            new_fp: Candidate fingerprint.

        Returns:
            Human-readable lesson string.
        """
        strategy = modification.get("strategy", "unknown")
        bit_set = set(changed_bits)

        # Categorize changed bits by functional region
        fp_len = len(new_fp)
        bit_ranges = {
            "hydrogen_bond_donor": set(range(0, min(200, fp_len))),
            "hydrophobic": set(range(200, min(600, fp_len))),
            "aromatic": set(range(600, min(900, fp_len))),
            "electrostatic": set(range(900, min(1100, fp_len))),
            "misc": set(range(min(1100, fp_len), fp_len)),
        }

        involved_regions = []
        for region_name, region_bits in bit_ranges.items():
            if bit_set & region_bits:
                involved_regions.append(region_name)

        region_str = (
            ", ".join(involved_regions) if involved_regions else "miscellaneous"
        )

        n_changed = len(changed_bits)
        bit_summary = ", ".join(str(b) for b in changed_bits[:5])
        if n_changed > 5:
            bit_summary += f" and {n_changed - 5} more"

        if new_affinity < old_affinity:
            lesson = (
                f"Flipping bits [{bit_summary}] using {strategy} reduced affinity "
                f"from {old_affinity:.2f} to {new_affinity:.2f} nM. "
                f"These bits likely encode favorable {region_str} interactions."
            )
        else:
            lesson = (
                f"Flipping bits [{bit_summary}] using {strategy} increased affinity "
                f"from {old_affinity:.2f} to {new_affinity:.2f} nM. "
                f"These modifications disrupt {region_str} binding features."
            )

        return lesson

    def _update_statistics(
        self,
        store: CognitionStore,
        changed_bits: List[Tuple[int, int]],
        improvement: float,
        new_fp: np.ndarray,
    ) -> None:
        """Update per-bit statistics in the cognition store.

        For each changed bit, records whether it was flipped up or down and
        accumulates the average improvement associated with that flip.

        Args:
            store: Cognition store to update.
            changed_bits: List of (bit_position, new_value) tuples.
            improvement: Affinity change (negative = improvement).
            new_fp: Candidate fingerprint.
        """
        for bit_pos, new_value in changed_bits:
            if bit_pos not in store.statistical_patterns:
                store.statistical_patterns[bit_pos] = {
                    "flip_up": 0,
                    "flip_down": 0,
                    "total_improvement": 0.0,
                    "observations": 0,
                    "avg_improvement": 0.0,
                }

            stats = store.statistical_patterns[bit_pos]

            if new_value == 1:
                stats["flip_up"] += 1
            else:
                stats["flip_down"] += 1

            stats["observations"] += 1
            stats["total_improvement"] += improvement
            stats["avg_improvement"] = (
                stats["total_improvement"] / stats["observations"]
            )

        logger.debug(
            "Updated statistics for %d bits (improvement=%+.3f)",
            len(changed_bits),
            improvement,
        )

    def _generate_new_smiles(self, parent_smiles: str, cycle_number: int) -> str:
        """Generate a new SMILES string representing the modified molecule.

        In a production system, this would use a graph-based generative model
        or RDKit to decode the fingerprint back to a valid molecule. Here we
        create a deterministic derivative marker for traceability.

        Args:
            parent_smiles: Original parent SMILES.
            cycle_number: Current cycle number.

        Returns:
            A new SMILES string (or derivative marker).
        """
        # In production: decode fp back to SMILES using a generative model
        # For mock: append cycle marker to create traceable derivative
        return f"{parent_smiles}_[C{cycle_number}]"
