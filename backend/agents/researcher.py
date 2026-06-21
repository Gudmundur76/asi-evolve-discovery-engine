"""Researcher agent for the ASI-Evolve molecular discovery engine.

The Researcher agent analyzes the cognition store's accumulated knowledge and
proposes the next molecular modification strategy. It selects from four
strategies (exploration, guided_mutation, bit_flip, crossover) based on the
current cycle number and the quality of statistical patterns available.
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from backend.agents.cognition_store import CognitionStore

logger = logging.getLogger(__name__)

# Strategy identifiers used throughout the system
STRATEGY_EXPLORATION = "exploration"
STRATEGY_GUIDED_MUTATION = "guided_mutation"
STRATEGY_BIT_FLIP = "bit_flip"
STRATEGY_CROSSOVER = "crossover"

# All available strategies
ALL_STRATEGIES = [
    STRATEGY_EXPLORATION,
    STRATEGY_GUIDED_MUTATION,
    STRATEGY_BIT_FLIP,
    STRATEGY_CROSSOVER,
]


class ResearcherAgent:
    """Agent that proposes molecular modifications based on accumulated knowledge.

    Strategy selection follows a phased approach:
    1. Cycles 1-5: Pure exploration to map the fitness landscape.
    2. Cycles 6+: Guided mutation using statistical patterns when available.
    3. Intermediate: Targeted single-bit flips for fine-tuning.
    4. Crossover: Combine features from multiple good candidates.
    """

    def __init__(self, cognition_store: CognitionStore) -> None:
        """Initialize the Researcher agent.

        Args:
            cognition_store: Shared knowledge store containing all past cycles.
        """
        self.store = cognition_store
        logger.debug("ResearcherAgent initialized (target=%s)", self.store.target_name)

    def propose_modification(
        self,
        current_best_smiles: str,
        current_best_fp: List[int],
        statistical_patterns: Optional[dict] = None,
        accumulated_lessons: Optional[List[str]] = None,
        cycle_number: int = 1,
    ) -> Dict[str, Any]:
        """Propose the next molecular modification strategy.

        Selects a strategy based on cycle number and pattern quality, then
        identifies specific target bits to modify.

        Args:
            current_best_smiles: SMILES of the current best molecule.
            current_best_fp: Sparse fingerprint of the current best molecule.
            statistical_patterns: Per-bit statistics (optional, uses store if None).
            accumulated_lessons: List of past lessons (optional, uses store if None).
            cycle_number: Current optimization cycle number.

        Returns:
            Dict with keys: strategy, target_bits, rationale, confidence.
        """
        patterns = statistical_patterns or self.store.statistical_patterns
        lessons = accumulated_lessons or self.store.accumulated_lessons

        strategy = self._select_strategy(cycle_number, patterns, lessons)
        target_bits = self._select_target_bits(
            strategy, current_best_fp, patterns, cycle_number
        )
        rationale = self._build_rationale(
            strategy, target_bits, patterns, lessons, cycle_number
        )
        confidence = self._estimate_confidence(strategy, patterns, cycle_number)

        modification = {
            "strategy": strategy,
            "target_bits": target_bits,
            "rationale": rationale,
            "confidence": round(confidence, 4),
        }

        logger.info(
            "Cycle %d | Strategy: %s | Bits: %s | Confidence: %.3f",
            cycle_number,
            strategy,
            target_bits[:5] if len(target_bits) > 5 else target_bits,
            confidence,
        )
        return modification

    # ------------------------------------------------------------------ #
    # Strategy selection
    # ------------------------------------------------------------------ #

    def _select_strategy(
        self, cycle_number: int, patterns: dict, lessons: List[str]
    ) -> str:
        """Select the most appropriate strategy for the current cycle.

        Args:
            cycle_number: Current optimization cycle number.
            patterns: Per-bit statistical patterns.
            lessons: Accumulated lessons list.

        Returns:
            Strategy name string.
        """
        # Phase 1: Exploration (cycles 1-5) — random exploration of the space
        if cycle_number <= 5:
            return STRATEGY_EXPLORATION

        # Phase 2: Check if we have meaningful patterns for guided mutation
        has_strong_patterns = self._has_strong_patterns(patterns)

        if has_strong_patterns and cycle_number % 3 != 0:
            return STRATEGY_GUIDED_MUTATION

        # Phase 3: Crossover when we have 2+ good candidates
        top_candidates = self.store.get_top_candidates(n=5)
        if len(top_candidates) >= 2 and cycle_number % 4 == 0:
            return STRATEGY_CROSSOVER

        # Phase 4: Targeted bit flip for fine-tuning
        if cycle_number % 3 == 0:
            return STRATEGY_BIT_FLIP

        # Default back to exploration if nothing else fits
        if not has_strong_patterns:
            return STRATEGY_EXPLORATION

        return STRATEGY_GUIDED_MUTATION

    def _has_strong_patterns(self, patterns: dict) -> bool:
        """Check whether the statistical patterns contain actionable signals.

        A 'strong' pattern means we have bits with at least 2 observations
        and a meaningful average improvement.

        Args:
            patterns: Per-bit statistics dict.

        Returns:
            True if actionable patterns exist.
        """
        if not patterns:
            return False
        for stats in patterns.values():
            observations = stats.get("flip_up", 0) + stats.get("flip_down", 0)
            if observations >= 2 and abs(stats.get("avg_improvement", 0.0)) > 0.1:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Target bit selection
    # ------------------------------------------------------------------ #

    def _select_target_bits(
        self,
        strategy: str,
        current_best_fp: List[int],
        patterns: dict,
        cycle_number: int,
    ) -> List[int]:
        """Select specific bit positions to modify.

        Args:
            strategy: Selected modification strategy.
            current_best_fp: Sparse fingerprint of current best.
            patterns: Per-bit statistics.
            cycle_number: Current cycle number.

        Returns:
            List of bit positions to target.
        """
        current_bits_set = set(current_best_fp)
        all_bits = list(range(2048))  # Default fingerprint size

        if strategy == STRATEGY_EXPLORATION:
            return self._bits_for_exploration(
                current_bits_set, all_bits, patterns, cycle_number
            )

        if strategy == STRATEGY_GUIDED_MUTATION:
            return self._bits_for_guided_mutation(current_bits_set, patterns)

        if strategy == STRATEGY_BIT_FLIP:
            return self._bits_for_bit_flip(current_bits_set, patterns)

        if strategy == STRATEGY_CROSSOVER:
            return self._bits_for_crossover(current_bits_set)

        # Fallback
        return self._bits_for_exploration(current_bits_set, all_bits, patterns, cycle_number)

    def _bits_for_exploration(
        self,
        current_bits_set: set,
        all_bits: List[int],
        patterns: dict,
        cycle_number: int,
        n_bits: int = 5,
    ) -> List[int]:
        """Pick random bits with slight bias toward less-explored positions.

        Args:
            current_bits_set: Currently set bit positions.
            all_bits: All available bit positions.
            patterns: Existing statistics (to find unexplored bits).
            cycle_number: Current cycle number.
            n_bits: Number of bits to flip.

        Returns:
            List of bit positions to modify.
        """
        explored = set()
        if patterns:
            for key in patterns:
                try:
                    explored.add(int(key))
                except (ValueError, TypeError):
                    explored.add(key)

        # Bias: less-explored bits get higher weight
        weights = []
        for b in all_bits:
            if b in explored:
                weights.append(0.3)  # Lower weight for explored
            else:
                weights.append(1.0)  # Higher weight for unexplored

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        selected = np_random_choice(all_bits, size=n_bits, replace=False, p=weights)
        return sorted(selected.tolist())

    def _bits_for_guided_mutation(
        self, current_bits_set: set, patterns: dict, n_bits: int = 3
    ) -> List[int]:
        """Use statistical patterns to pick bits with best improvement history.

        Bits are weighted by their avg_improvement; we flip bits that have
        historically produced the largest affinity reductions.

        Args:
            current_bits_set: Currently set bit positions.
            patterns: Per-bit statistics.
            n_bits: Number of bits to mutate.

        Returns:
            List of bit positions to modify.
        """
        if not patterns:
            # Fallback to exploration
            return self._bits_for_exploration(
                current_bits_set, list(range(2048)), patterns, 0
            )

        # Sort bits by avg_improvement (ascending = most negative = best)
        sorted_bits = sorted(
            patterns.items(),
            key=lambda item: item[1].get("avg_improvement", 0.0),
        )

        # Pick top N bits that are in the current fingerprint
        # (we want to flip them to potentially improve)
        target_bits = []
        for bit_key, stats in sorted_bits:
            try:
                bit_pos = int(bit_key)
            except (ValueError, TypeError):
                bit_pos = bit_key
            target_bits.append(bit_pos)
            if len(target_bits) >= n_bits:
                break

        if len(target_bits) < n_bits:
            # Fill with random bits
            remaining = [b for b in range(2048) if b not in target_bits]
            needed = n_bits - len(target_bits)
            if remaining and needed > 0:
                extra = random.sample(remaining, min(needed, len(remaining)))
                target_bits.extend(extra)

        return sorted(target_bits[:n_bits])

    def _bits_for_bit_flip(self, current_bits_set: set, patterns: dict) -> List[int]:
        """Select a single high-value bit for targeted flipping.

        Args:
            current_bits_set: Currently set bit positions.
            patterns: Per-bit statistics.

        Returns:
            List containing a single bit position.
        """
        if patterns:
            # Pick the single bit with best historical improvement
            sorted_bits = sorted(
                patterns.items(),
                key=lambda item: item[1].get("avg_improvement", 0.0),
            )
            for bit_key, _ in sorted_bits:
                try:
                    bit_pos = int(bit_key)
                except (ValueError, TypeError):
                    bit_pos = bit_key
                return [bit_pos]

        # Fallback: random single bit
        return [random.randint(0, 2047)]

    def _bits_for_crossover(self, current_bits_set: set) -> List[int]:
        """Select bits for crossover by identifying alternative candidate bits.

        For crossover we return a set of bits that represent the 'other'
        candidate's distinguishing features. The engineer will use these
        alongside the current best fingerprint.

        Args:
            current_bits_set: Currently set bit positions in the best candidate.

        Returns:
            List of bit positions representing crossover targets.
        """
        top = self.store.get_top_candidates(n=2)
        if len(top) < 2:
            # Not enough candidates — fallback
            return self._bits_for_exploration(
                current_bits_set, list(range(2048)), {}, 0, n_bits=4
            )

        # Bits present in candidate 2 but not in candidate 1
        best_fp = set(top[0].new_fp)
        second_fp = set(top[1].new_fp)
        diff_bits = list(second_fp - best_fp)

        if not diff_bits:
            return self._bits_for_exploration(
                current_bits_set, list(range(2048)), {}, 0, n_bits=4
            )

        # Return up to 5 distinguishing bits
        return sorted(diff_bits[:5])

    # ------------------------------------------------------------------ #
    # Rationale generation
    # ------------------------------------------------------------------ #

    def _build_rationale(
        self,
        strategy: str,
        target_bits: List[int],
        patterns: dict,
        lessons: List[str],
        cycle_number: int,
    ) -> str:
        """Build a human-readable explanation for the chosen strategy.

        Args:
            strategy: Selected strategy name.
            target_bits: Bit positions being modified.
            patterns: Per-bit statistics.
            lessons: Accumulated lessons.
            cycle_number: Current cycle number.

        Returns:
            Human-readable rationale string.
        """
        bit_str = ", ".join(str(b) for b in target_bits[:10])
        if len(target_bits) > 10:
            bit_str += f", ... ({len(target_bits)} total)"

        if strategy == STRATEGY_EXPLORATION:
            return (
                f"Cycle {cycle_number}: Early exploration phase. "
                f"Randomly probing bits [{bit_str}] to map the affinity landscape "
                f"and gather initial statistics."
            )

        if strategy == STRATEGY_GUIDED_MUTATION:
            # Find the avg_improvement for target bits
            imp_values = []
            for b in target_bits:
                stats = patterns.get(b) or patterns.get(str(b), {})
                imp_values.append(stats.get("avg_improvement", 0.0))
            avg_imp = sum(imp_values) / len(imp_values) if imp_values else 0.0
            return (
                f"Cycle {cycle_number}: Using guided mutation based on accumulated "
                f"patterns. Target bits [{bit_str}] have historical avg improvement "
                f"of {avg_imp:.3f} nM. Flipping these bits has correlated with "
                f"affinity improvements in past cycles."
            )

        if strategy == STRATEGY_BIT_FLIP:
            stats = patterns.get(target_bits[0]) if patterns else {}
            if not stats and target_bits:
                stats = patterns.get(str(target_bits[0]), {})
            avg_imp = stats.get("avg_improvement", 0.0) if stats else 0.0
            return (
                f"Cycle {cycle_number}: Targeted single-bit flip at position "
                f"{target_bits[0]}. This bit has historical avg improvement of "
                f"{avg_imp:.3f} nM. Fine-tuning the most impactful individual bit."
            )

        if strategy == STRATEGY_CROSSOVER:
            top = self.store.get_top_candidates(n=2)
            c1_aff = top[0].predicted_affinity_nm if top else 0.0
            c2_aff = top[1].predicted_affinity_nm if len(top) > 1 else 0.0
            return (
                f"Cycle {cycle_number}: Crossover between top candidates "
                f"(affinities: {c1_aff:.2f} and {c2_aff:.2f} nM). "
                f"Combining distinguishing bits [{bit_str}] from the second-best "
                f"candidate into the current best."
            )

        return f"Cycle {cycle_number}: Default strategy with bits [{bit_str}]."

    def _estimate_confidence(
        self, strategy: str, patterns: dict, cycle_number: int
    ) -> float:
        """Estimate confidence in the proposed modification.

        Confidence increases with more cycles and stronger patterns.

        Args:
            strategy: Selected strategy name.
            patterns: Per-bit statistics.
            cycle_number: Current cycle number.

        Returns:
            Confidence float in [0, 1].
        """
        base_confidence = 0.3

        # More cycles = more confidence (up to a point)
        cycle_bonus = min(cycle_number / 20.0, 0.3)

        # Strong patterns = more confidence
        pattern_bonus = 0.0
        if patterns:
            n_patterned_bits = len([b for b in patterns.values() if b.get("flip_up", 0) + b.get("flip_down", 0) >= 2])
            pattern_bonus = min(n_patterned_bits / 50.0, 0.3)

        # Strategy-specific adjustments
        strategy_multiplier = 1.0
        if strategy == STRATEGY_GUIDED_MUTATION:
            strategy_multiplier = 1.2
        elif strategy == STRATEGY_EXPLORATION:
            strategy_multiplier = 0.8

        confidence = (base_confidence + cycle_bonus + pattern_bonus) * strategy_multiplier
        return min(confidence, 0.95)  # Cap at 0.95 — never fully certain


# Small helper since we can't assume numpy is available for random choice
import numpy as np


def np_random_choice(a: List[int], size: int, replace: bool = False, p: Optional[List[float]] = None):
    """Wrapper around numpy.random.choice for type compatibility."""
    return np.random.choice(a, size=size, replace=replace, p=p)
