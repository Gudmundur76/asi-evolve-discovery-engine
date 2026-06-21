"""
Discovery validation orchestrator.

:class:`DiscoveryValidator` ties together molecular docking (``VinaDocker``)
and ADMET profiling (``SwissADMEClient``) to produce a single, authoritative
validation result for each candidate compound.

The confidence-score formula is::

    confidence = affinity_norm + docking_norm + admet_contrib

where each term is bounded so the total lies in [0, 1].
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.validation.vina_docker import VinaDocker
from backend.validation.swissadme_client import SwissADMEClient

logger = logging.getLogger(__name__)

# Thresholds (match the specification)
_DOCKING_SCORE_PASS = -9.0  # kcal/mol
_DOCKING_SCORE_WORST = -5.0  # kcal/mol (used for normalisation)
_DOCKING_SCORE_BEST = -13.0  # kcal/mol (used for normalisation)

# Weighting coefficients
_W_AFFINITY = 0.4
_W_DOCKING = 0.3
_W_ADMET = 0.3


class DiscoveryValidator:
    """Orchestrate docking + ADMET validation for a single candidate.

    Parameters
    ----------
    receptor_path:
        Path to the receptor PDBQT file forwarded to ``VinaDocker``.
    center, box_size:
        Search-space geometry forwarded to ``VinaDocker``.
    """

    def __init__(
        self,
        receptor_path: str,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box_size: float = 20.0,
    ) -> None:
        self.dock_engine = VinaDocker(
            receptor_path=receptor_path,
            center=center,
            box_size=box_size,
        )
        self.admet_engine = SwissADMEClient()
        self._db_mock: list[dict[str, Any]] = []  # in-memory DB mock

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_candidate(
        self,
        smiles: str,
        predicted_affinity: float,
        candidate_id: str,
        target_chembl_id: str = "",
    ) -> dict[str, Any]:
        """Run the full validation pipeline on a single candidate.

        Pipeline steps:
            1. Ligand preparation (SMILES -> PDBQT)
            2. Molecular docking via Vina (or mock)
            3. ADMET profiling via SwissADME (or local fallback)
            4. Threshold checks and confidence scoring
            5. (Mock) database persistence
            6. Evidence PDF path generation

        Parameters
        ----------
        smiles:
            Candidate molecule SMILES.
        predicted_affinity:
            Model-predicted pIC50 / pKi value (higher = tighter binding).
        candidate_id:
            Unique identifier for this candidate.
        target_chembl_id:
            Optional ChEMBL target identifier for grouping.

        Returns
        -------
        dict
            Complete validation result dictionary.
        """
        result: dict[str, Any] = {
            "candidate_id": candidate_id,
            "smiles": smiles,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target_chembl_id": target_chembl_id,
            "docking": None,
            "admet": None,
            "docking_pass": False,
            "admet_pass": False,
            "overall_pass": False,
            "confidence_score": 0.0,
            "db_record_id": None,
            "evidence_pdf_path": None,
            "errors": [],
        }

        # 1. Ligand preparation
        try:
            ligand_pdbqt = self.dock_engine.prepare_ligand(smiles)
        except Exception as exc:
            logger.error("Ligand preparation failed: %s", exc, exc_info=True)
            result["errors"].append(f"ligand_preparation: {exc}")
            ligand_pdbqt = None

        # 2. Docking
        if ligand_pdbqt:
            try:
                docking_result = self.dock_engine.dock(ligand_pdbqt)
                result["docking"] = docking_result
                result["docking_pass"] = self._check_docking_pass(docking_result)
            except Exception as exc:
                logger.error("Docking failed: %s", exc, exc_info=True)
                result["errors"].append(f"docking: {exc}")

        # 3. ADMET profiling
        try:
            admet_result = self.admet_engine.profile(smiles)
            result["admet"] = admet_result
            result["admet_pass"] = bool(
                admet_result.get("is_druglike", False)
                and admet_result.get("overall_pass", False)
            )
        except Exception as exc:
            logger.error("ADMET profiling failed: %s", exc, exc_info=True)
            result["errors"].append(f"admet: {exc}")

        # 4. Confidence score
        result["confidence_score"] = self._compute_confidence(
            predicted_affinity=predicted_affinity,
            docking_result=result["docking"],
            admet_pass=result["admet_pass"],
        )

        # 5. Overall pass
        result["overall_pass"] = result["docking_pass"] and result["admet_pass"]

        # 6. Mock database persistence
        try:
            db_record = self._mock_db_insert(result)
            result["db_record_id"] = db_record.get("id")
        except Exception as exc:
            logger.error("DB mock insert failed: %s", exc, exc_info=True)
            result["errors"].append(f"db_insert: {exc}")

        # 7. Evidence PDF path
        evidence_path = self._evidence_path(candidate_id)
        result["evidence_pdf_path"] = str(evidence_path)

        logger.info(
            "Validation complete for %s: overall=%s confidence=%.3f",
            candidate_id,
            result["overall_pass"],
            result["confidence_score"],
        )
        return result

    # ------------------------------------------------------------------
    # Threshold & scoring helpers
    # ------------------------------------------------------------------

    def _check_docking_pass(self, docking_result: dict[str, Any] | None) -> bool:
        """Return True if the best docking score meets the pass threshold."""
        if docking_result is None:
            return False
        score = docking_result.get("docking_score", 0.0)
        return score <= _DOCKING_SCORE_PASS

    def _compute_confidence(
        self,
        predicted_affinity: float,
        docking_result: dict[str, Any] | None,
        admet_pass: bool,
    ) -> float:
        """Compute composite confidence score in [0, 1].

        Terms
        -----
        * Predicted affinity contributes up to 0.4 (higher pIC50 -> higher score)
        * Docking score contributes up to 0.3 (more negative -> higher score)
        * ADMET pass contributes 0.3 (all-or-nothing)
        """
        # Normalise predicted affinity (assume range 4--12 pIC50)
        affinity_norm = self._clamp(
            (predicted_affinity - 4.0) / 8.0, 0.0, 1.0
        ) * _W_AFFINITY

        # Normalise docking score (more negative = better)
        if docking_result and docking_result.get("success"):
            raw_score = docking_result.get("docking_score", 0.0)
            docking_norm = self._clamp(
                (raw_score - _DOCKING_SCORE_WORST)
                / (_DOCKING_SCORE_BEST - _DOCKING_SCORE_WORST),
                0.0,
                1.0,
            ) * _W_DOCKING
        else:
            docking_norm = 0.0

        # ADMET contribution
        admet_contrib = _W_ADMET if admet_pass else 0.0

        return round(affinity_norm + docking_norm + admet_contrib, 4)

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        """Constrain *value* to the inclusive interval [*lo*, *hi*]."""
        return float(max(lo, min(hi, value)))

    # ------------------------------------------------------------------
    # Mock database helpers
    # ------------------------------------------------------------------

    def _mock_db_insert(self, result: dict[str, Any]) -> dict[str, Any]:
        """Simulate a database insert by storing in a list and returning an ID.

        In production this would call ``backend.database.discovery_db.create_discovery``.
        """
        record_id = len(self._db_mock) + 1
        record = {
            "id": record_id,
            "candidate_id": result["candidate_id"],
            "smiles": result["smiles"],
            "timestamp": result["timestamp"],
            "target_chembl_id": result["target_chembl_id"],
            "docking_score": (
                result["docking"].get("docking_score", 0.0)
                if result["docking"]
                else None
            ),
            "docking_pass": result["docking_pass"],
            "admet_pass": result["admet_pass"],
            "overall_pass": result["overall_pass"],
            "confidence_score": result["confidence_score"],
            "admet_summary": (
                {
                    "mw": result["admet"].get("mw"),
                    "logp": result["admet"].get("logp"),
                    "tpsa": result["admet"].get("tpsa"),
                    "lipinski_violations": result["admet"].get("lipinski_violations"),
                    "is_druglike": result["admet"].get("is_druglike"),
                    "overall_pass": result["admet"].get("overall_pass"),
                }
                if result["admet"]
                else None
            ),
            "evidence_pdf_path": result["evidence_pdf_path"],
        }
        self._db_mock.append(record)
        logger.debug("Mock DB insert: id=%d candidate=%s", record_id, result["candidate_id"])
        return record

    def _evidence_path(self, candidate_id: str) -> Path:
        """Return a deterministic PDF path for the candidate evidence document."""
        evidence_dir = Path("data/evidence")
        evidence_dir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", candidate_id)
        return evidence_dir / f"{safe_id}_evidence.pdf"

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_mock_db_records(self) -> list[dict[str, Any]]:
        """Return all mock DB records (useful for testing)."""
        return list(self._db_mock)

    def clear_mock_db(self) -> None:
        """Clear the mock database (useful between test runs)."""
        self._db_mock.clear()
