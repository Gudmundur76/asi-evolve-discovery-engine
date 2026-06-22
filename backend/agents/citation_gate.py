"""Citation Gate — Gate 1 in the HIV-1 Protease discovery pipeline.

Before a candidate enters the ADMET + docking validation pipeline, it must
pass the citation gate: a claim about its predicted binding affinity is
submitted to citation.manus.space (ttruthdesk-platform) and must be returned
as 'Supported' with confidence ≥ 0.85.

This gate ensures every corpus-accepted candidate has real PubMed-backed
evidence for its mechanism of action, not just a model prediction.

Architecture:
    novus-is / asi-evolve-discovery-engine (generator)
        │
        │  POST /api/public/verify-claim
        ▼
    citation.manus.space  ←── ttruthdesk-platform (backend)
        │                         │
        │                    8-stage pipeline:
        │                    extract → PDB → PubMed → UniProt
        │                    → friction → citation chain
        │                    → composite truth → graph edges
        │
        └── verdict + confidence + PMIDs + SPO + contradictions
        │
        ▼
    Gate 1 decision: PASS / FAIL
        │
        ▼
    ADMET + Vina (Gate 2 + Gate 3)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

CITATION_API_URL = "https://citation.manus.space/api/public/verify-claim"
CITATION_VERTICAL = "hiv_protease"
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
REQUEST_TIMEOUT = 15  # seconds
MAX_RETRIES = 2
RETRY_DELAY = 2.0  # seconds


@dataclass
class CitationResult:
    """Result from the citation gate for a single candidate."""
    smiles: str
    claim: str
    verdict: str                        # "Supported" | "Contradicted" | "Insufficient Evidence"
    confidence_score: float
    pubmed_ids: List[str] = field(default_factory=list)
    spo_subject: Optional[str] = None
    spo_predicate: Optional[str] = None
    spo_object: Optional[str] = None
    contradictions: List[str] = field(default_factory=list)
    rationale: str = ""
    gate_passed: bool = False
    error: Optional[str] = None
    latency_ms: float = 0.0
    raw_response: Dict[str, Any] = field(default_factory=dict)


class CitationGate:
    """Gate 1: citation.manus.space verification.

    Submits a natural-language claim about a candidate molecule to the
    ttruthdesk-platform verification API and returns a structured result.

    Parameters
    ----------
    confidence_threshold:
        Minimum confidence score required to pass the gate (default 0.85).
    api_url:
        citation.manus.space API endpoint.
    vertical:
        Domain vertical for the claim (default 'hiv_protease').
    """

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        api_url: str = CITATION_API_URL,
        vertical: str = CITATION_VERTICAL,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.api_url = api_url
        self.vertical = vertical
        logger.info(
            "CitationGate initialized: threshold=%.2f, url=%s, vertical=%s",
            confidence_threshold, api_url, vertical,
        )

    def verify(
        self,
        smiles: str,
        compound_name: str,
        predicted_pic50: float,
        target_name: str = "HIV-1 protease",
        quantum_score: float = 0.5,
    ) -> CitationResult:
        """Verify a single candidate against the citation knowledge graph.

        Parameters
        ----------
        smiles:
            SMILES string of the candidate molecule.
        compound_name:
            Human-readable name or ID for the candidate.
        predicted_pic50:
            Predicted pIC50 value from the ensemble model.
        target_name:
            Target protein name (default 'HIV-1 protease').
        quantum_score:
            VQE-derived quantum affinity score in [0, 1] from QuantumPredictor.
            Included in the claim for richer citation context (default 0.5).

        Returns
        -------
        CitationResult with gate_passed=True if verdict is 'Supported'
        and confidence_score >= threshold.
        """
        claim = self._build_claim(compound_name, predicted_pic50, target_name, quantum_score)
        logger.debug("CitationGate: verifying claim: %s", claim)

        for attempt in range(MAX_RETRIES + 1):
            try:
                t0 = time.time()
                response = requests.post(
                    self.api_url,
                    json={"claim": claim, "vertical": self.vertical},
                    timeout=REQUEST_TIMEOUT,
                )
                latency_ms = (time.time() - t0) * 1000

                if response.status_code != 200:
                    logger.warning(
                        "CitationGate HTTP %d for %s (attempt %d/%d)",
                        response.status_code, compound_name, attempt + 1, MAX_RETRIES + 1,
                    )
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                        continue
                    return CitationResult(
                        smiles=smiles,
                        claim=claim,
                        verdict="Insufficient Evidence",
                        confidence_score=0.0,
                        gate_passed=False,
                        error=f"HTTP {response.status_code}",
                        latency_ms=latency_ms,
                    )

                data = response.json()
                return self._parse_response(smiles, claim, data, latency_ms)

            except requests.Timeout:
                logger.warning(
                    "CitationGate timeout for %s (attempt %d/%d)",
                    compound_name, attempt + 1, MAX_RETRIES + 1,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                return CitationResult(
                    smiles=smiles,
                    claim=claim,
                    verdict="Insufficient Evidence",
                    confidence_score=0.0,
                    gate_passed=False,
                    error="Request timeout",
                )
            except Exception as exc:
                logger.error(
                    "CitationGate error for %s: %s", compound_name, exc
                )
                return CitationResult(
                    smiles=smiles,
                    claim=claim,
                    verdict="Insufficient Evidence",
                    confidence_score=0.0,
                    gate_passed=False,
                    error=str(exc),
                )

        # Should not reach here
        return CitationResult(
            smiles=smiles,
            claim=claim,
            verdict="Insufficient Evidence",
            confidence_score=0.0,
            gate_passed=False,
            error="Max retries exceeded",
        )

    def verify_batch(
        self,
        candidates: List[Dict[str, Any]],
        target_name: str = "HIV-1 protease",
    ) -> List[CitationResult]:
        """Verify a batch of candidates.

        Parameters
        ----------
        candidates:
            List of dicts with keys: smiles, name, predicted_pic50.
        target_name:
            Target protein name.

        Returns
        -------
        List of CitationResult, one per candidate.
        """
        results = []
        passed = 0
        for i, cand in enumerate(candidates):
            result = self.verify(
                smiles=cand.get("smiles", ""),
                compound_name=cand.get("name", f"candidate_{i:04d}"),
                predicted_pic50=cand.get("predicted_pic50", 0.0),
                target_name=target_name,
            )
            results.append(result)
            if result.gate_passed:
                passed += 1

        logger.info(
            "CitationGate batch: %d/%d passed (threshold=%.2f)",
            passed, len(candidates), self.confidence_threshold,
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_claim(
        self,
        compound_name: str,
        predicted_pic50: float,
        target_name: str,
        quantum_score: float = 0.5,
    ) -> str:
        """Build a natural-language claim for the verification API."""
        # Convert pIC50 to IC50 in nM for a more natural claim
        ic50_nm = 10 ** (9 - predicted_pic50) if predicted_pic50 > 0 else 1000.0
        quantum_note = (
            f" Quantum VQE affinity score: {quantum_score:.3f} (Origin Pilot / WuKong)."
            if quantum_score != 0.5
            else ""
        )
        return (
            f"{compound_name} inhibits {target_name} with predicted pIC50 "
            f"{predicted_pic50:.2f} (IC50 ≈ {ic50_nm:.1f} nM), consistent with "
            f"the hydroxyethylamine scaffold class of HIV protease inhibitors."
            f"{quantum_note}"
        )

    def _parse_response(
        self,
        smiles: str,
        claim: str,
        data: Dict[str, Any],
        latency_ms: float,
    ) -> CitationResult:
        """Parse the citation.manus.space API response into a CitationResult."""
        verdict = data.get("verdict", "Insufficient Evidence")
        confidence = float(data.get("confidenceScore", 0.0))
        gate_passed = (
            verdict == "Supported"
            and confidence >= self.confidence_threshold
        )

        # Extract PubMed IDs from pubmedResults list
        pubmed_ids: List[str] = []
        for item in data.get("pubmedResults", []):
            if isinstance(item, dict):
                pmid = item.get("pmid") or item.get("id")
                if pmid:
                    pubmed_ids.append(str(pmid))
            elif isinstance(item, str):
                pubmed_ids.append(item)

        # Extract SPO triple
        spo = data.get("spo", {})
        if isinstance(spo, dict):
            spo_subject = spo.get("subject")
            spo_predicate = spo.get("predicate")
            spo_object = spo.get("object")
        else:
            spo_subject = spo_predicate = spo_object = None

        # Extract contradictions
        contradictions = data.get("contradictions", [])
        if isinstance(contradictions, list):
            contradictions = [str(c) for c in contradictions]

        result = CitationResult(
            smiles=smiles,
            claim=claim,
            verdict=verdict,
            confidence_score=confidence,
            pubmed_ids=pubmed_ids,
            spo_subject=spo_subject,
            spo_predicate=spo_predicate,
            spo_object=spo_object,
            contradictions=contradictions,
            rationale=data.get("rationale", ""),
            gate_passed=gate_passed,
            latency_ms=latency_ms,
            raw_response=data,
        )

        logger.info(
            "CitationGate: verdict=%s, confidence=%.3f, gate=%s, pmids=%d, latency=%.0fms",
            verdict, confidence, "PASS" if gate_passed else "FAIL",
            len(pubmed_ids), latency_ms,
        )
        return result
