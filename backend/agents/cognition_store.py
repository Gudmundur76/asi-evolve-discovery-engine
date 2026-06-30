"""Cognition store for the ASI-Evolve molecular discovery engine.

This module defines the data structures and persistence layer for recording
molecular optimization cycles, tracking the best candidates discovered, and
accumulating statistical patterns about which fingerprint modifications lead
to improved binding affinity.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class _CognitionJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime and numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return {"__datetime__": True, "iso": obj.isoformat()}
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def _cognition_json_decoder(obj: dict) -> Any:
    """Object hook for decoding cognition store JSON (datetimes)."""
    if obj.get("__datetime__"):
        return datetime.fromisoformat(obj["iso"])
    return obj


@dataclass
class CycleRecord:
    """A single cycle record in the molecular optimization loop.

    Attributes:
        cycle_id: Sequential identifier for this optimization cycle.
        timestamp: When the cycle was executed.
        parent_smiles: SMILES of the starting molecule for this cycle.
        proposed_modification: Dict with keys strategy, target_bits, rationale, confidence.
        new_smiles: SMILES of the proposed modified molecule.
        new_fp: Sparse fingerprint representation (list of set bit positions).
        predicted_affinity_nm: Predicted binding affinity in nanomolar.
        improvement: Change in affinity vs parent (negative = improvement).
        is_best_so_far: Whether this is the best affinity achieved to date.
        lesson: Human-readable lesson distilled from this cycle.
        fingerprint_diff: Bit positions that changed relative to parent.
    """

    cycle_id: int
    timestamp: datetime
    parent_smiles: str
    proposed_modification: dict
    new_smiles: str
    new_fp: List[int]
    predicted_affinity_nm: float
    improvement: float
    is_best_so_far: bool
    lesson: str
    fingerprint_diff: List[int]
    # Quantum provenance fields (optional — populated when quantum scoring is active)
    pic50_vqe: Optional[float] = None          # pIC50 derived from VQE score
    quantum_hardware: Optional[str] = None     # e.g. "WuKong (superconducting, 72 qubits)"
    provenance_status: Optional[str] = None    # "quantum-hardware" | "quantum-sim" | None
    confidence: Optional[float] = None         # ensemble confidence [0, 1]
    citation_ids: List[str] = field(default_factory=list)  # citation.is permanent URLs


@dataclass
class CognitionStore:
    """Persistent store for molecular optimization knowledge.

    Records every optimization cycle, tracks the best candidates, accumulates
    human-readable lessons, and maintains statistical patterns about which
    fingerprint bit flips correlate with affinity improvements.

    Attributes:
        target_chembl_id: ChEMBL identifier for the biological target.
        target_name: Human-readable target name.
        created_at: When the cognition store was created.
        cycles: Chronological list of all optimization cycles.
        best_affinity_ever: Best (lowest) affinity achieved so far (nM).
        best_smiles_ever: SMILES of the best-scoring molecule.
        best_fp_ever: Sparse fingerprint of the best molecule.
        accumulated_lessons: Extracted lessons from all cycles.
        statistical_patterns: Per-bit statistics from observed modifications.
    """

    target_chembl_id: str
    target_name: str
    created_at: datetime
    cycles: List[CycleRecord] = field(default_factory=list)
    best_affinity_ever: float = field(default=float("inf"))
    best_smiles_ever: str = field(default="")
    best_fp_ever: List[int] = field(default_factory=list)
    accumulated_lessons: List[str] = field(default_factory=list)
    statistical_patterns: dict = field(default_factory=dict)
    # Maps cycle_id → citation.is permanent URL for verified candidates
    citation_registry: Dict[int, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Core mutation helpers
    # ------------------------------------------------------------------ #

    def add_cycle(self, record: CycleRecord) -> None:
        """Append a cycle record and update running bests.

        Args:
            record: The completed cycle record to store.
        """
        self.cycles.append(record)
        self.accumulated_lessons.append(record.lesson)

        if record.predicted_affinity_nm < self.best_affinity_ever:
            self.best_affinity_ever = record.predicted_affinity_nm
            self.best_smiles_ever = record.new_smiles
            self.best_fp_ever = list(record.new_fp)
            logger.info(
                "New best affinity: %.3f nM (SMILES: %s)",
                self.best_affinity_ever,
                self.best_smiles_ever[:60],
            )

        logger.debug("Added cycle %d (affinity=%.3f nM)", record.cycle_id, record.predicted_affinity_nm)

    def get_lessons_summary(self, n_recent: int = 10) -> str:
        """Build a formatted summary of recent lessons for the Researcher agent.

        The summary includes:
        * Best candidates discovered so far
        * Which modification strategies worked
        * Which fingerprint bits were most valuable

        Args:
            n_recent: Number of recent cycles to include in the summary.

        Returns:
            A formatted, human-readable string.
        """
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("COGNITION STORE LESSONS SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Target: {self.target_name} ({self.target_chembl_id})")
        lines.append(f"Total cycles: {len(self.cycles)}")
        lines.append(f"Best affinity ever: {self.best_affinity_ever:.3f} nM")
        lines.append(f"Best SMILES: {self.best_smiles_ever[:80]}")
        lines.append("")

        # Top candidates
        lines.append("-" * 40)
        lines.append("TOP CANDIDATES")
        lines.append("-" * 40)
        for idx, cand in enumerate(self.get_top_candidates(n=5), 1):
            lines.append(
                f"  {idx}. Cycle {cand.cycle_id}: {cand.predicted_affinity_nm:.3f} nM "
                f"(strategy={cand.proposed_modification.get('strategy', 'N/A')})"
            )
        lines.append("")

        # Recent lessons
        lines.append("-" * 40)
        lines.append(f"RECENT LESSONS (last {n_recent})")
        lines.append("-" * 40)
        recent = self.accumulated_lessons[-n_recent:] if self.accumulated_lessons else []
        for i, lesson in enumerate(recent, 1):
            lines.append(f"  {i}. {lesson}")
        lines.append("")

        # Bit statistics
        lines.append("-" * 40)
        lines.append("VALUABLE BITS (sorted by avg improvement)")
        lines.append("-" * 40)
        bit_stats = self.get_bit_statistics()
        for bit, stats in list(bit_stats.items())[:10]:
            avg_imp = stats.get("avg_improvement", 0.0)
            flips_up = stats.get("flip_up", 0)
            flips_down = stats.get("flip_down", 0)
            lines.append(
                f"  Bit {bit}: avg_improvement={avg_imp:+.4f}, "
                f"flips_up={flips_up}, flips_down={flips_down}"
            )
        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def get_top_candidates(self, n: int = 5) -> List[CycleRecord]:
        """Return the top N candidates sorted by predicted affinity (ascending = best).

        Args:
            n: Maximum number of candidates to return.

        Returns:
            List of CycleRecord objects, sorted by affinity (best first).
        """
        sorted_cycles = sorted(self.cycles, key=lambda c: c.predicted_affinity_nm)
        return sorted_cycles[:n]

    def get_bit_statistics(self) -> Dict[str, dict]:
        """Return statistical patterns sorted by average improvement (descending).

        Returns:
            Dictionary mapping bit position strings to their statistics dicts,
            sorted by avg_improvement in descending order.
        """
        # Ensure all entries have avg_improvement calculated
        sorted_items = sorted(
            self.statistical_patterns.items(),
            key=lambda item: item[1].get("avg_improvement", 0.0),
            reverse=True,
        )
        return {str(k): v for k, v in sorted_items}

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Convert the entire cognition store to a plain dictionary.

        Returns:
            dict suitable for JSON serialization.
        """
        return {
            "target_chembl_id": self.target_chembl_id,
            "target_name": self.target_name,
            "created_at": self.created_at,
            "cycles": [
                {
                    "cycle_id": c.cycle_id,
                    "timestamp": c.timestamp,
                    "parent_smiles": c.parent_smiles,
                    "proposed_modification": c.proposed_modification,
                    "new_smiles": c.new_smiles,
                    "new_fp": c.new_fp,
                    "predicted_affinity_nm": c.predicted_affinity_nm,
                    "improvement": c.improvement,
                    "is_best_so_far": c.is_best_so_far,
                    "lesson": c.lesson,
                    "fingerprint_diff": c.fingerprint_diff,
                    "pic50_vqe": c.pic50_vqe,
                    "quantum_hardware": c.quantum_hardware,
                    "provenance_status": c.provenance_status,
                    "confidence": c.confidence,
                    "citation_ids": c.citation_ids,
                }
                for c in self.cycles
            ],
            "best_affinity_ever": self.best_affinity_ever,
            "best_smiles_ever": self.best_smiles_ever,
            "best_fp_ever": self.best_fp_ever,
            "accumulated_lessons": self.accumulated_lessons,
            "statistical_patterns": {
                str(k): v for k, v in self.statistical_patterns.items()
            },
            "citation_registry": {
                str(k): v for k, v in self.citation_registry.items()
            },
        }

    def save(self, path: str) -> None:
        """Serialize the cognition store to JSON.

        Uses a custom encoder to handle datetime and numpy types.

        Args:
            path: Destination file path.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(data, fh, cls=_CognitionJSONEncoder, indent=2)
        logger.info("Cognition store saved to %s (%d cycles)", path, len(self.cycles))

    @classmethod
    def load(cls, path: str) -> "CognitionStore":
        """Deserialize a cognition store from JSON.

        Args:
            path: Source file path.

        Returns:
            A fully reconstructed CognitionStore instance.
        """
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Cognition store not found: {path}")

        with open(src, "r", encoding="utf-8") as fh:
            data = json.load(fh, object_hook=_cognition_json_decoder)

        # Rebuild CycleRecord objects
        cycles = []
        for c_data in data.get("cycles", []):
            cycles.append(
                CycleRecord(
                    cycle_id=c_data["cycle_id"],
                    timestamp=c_data["timestamp"],
                    parent_smiles=c_data["parent_smiles"],
                    proposed_modification=c_data["proposed_modification"],
                    new_smiles=c_data["new_smiles"],
                    new_fp=c_data["new_fp"],
                    predicted_affinity_nm=c_data["predicted_affinity_nm"],
                    improvement=c_data["improvement"],
                    is_best_so_far=c_data["is_best_so_far"],
                    lesson=c_data["lesson"],
                    fingerprint_diff=c_data["fingerprint_diff"],
                    pic50_vqe=c_data.get("pic50_vqe"),
                    quantum_hardware=c_data.get("quantum_hardware"),
                    provenance_status=c_data.get("provenance_status"),
                    confidence=c_data.get("confidence"),
                    citation_ids=c_data.get("citation_ids", []),
                )
            )

        # Restore statistical_patterns with integer keys
        raw_patterns = data.get("statistical_patterns", {})
        restored_patterns: dict = {}
        for k, v in raw_patterns.items():
            restored_patterns[int(k)] = v

        store = cls(
            target_chembl_id=data["target_chembl_id"],
            target_name=data["target_name"],
            created_at=data["created_at"],
            cycles=cycles,
            best_affinity_ever=data.get("best_affinity_ever", float("inf")),
            best_smiles_ever=data.get("best_smiles_ever", ""),
            best_fp_ever=data.get("best_fp_ever", []),
            accumulated_lessons=data.get("accumulated_lessons", []),
            statistical_patterns=restored_patterns,
            citation_registry={
                int(k): v
                for k, v in data.get("citation_registry", {}).items()
            },
        )
        logger.info(
            "Cognition store loaded from %s (%d cycles)", path, len(store.cycles)
        )
        return store
