"""Loop scheduler for the ASI-Evolve molecular discovery engine.

The LoopScheduler orchestrates the three-agent optimization loop:
1. Researcher proposes a modification strategy.
2. Engineer applies the modification to a molecular fingerprint.
3. Analyzer evaluates the candidate and updates the cognition store.

The scheduler supports both single-cycle execution and a continuous async
loop that runs at a configurable interval (default: 72 minutes for 20 cycles/day).
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from backend.agents.analyzer import AnalyzerAgent
from backend.agents.cognition_store import CognitionStore
from backend.agents.engineer import EngineerAgent
from backend.agents.researcher import ResearcherAgent
from backend.config import settings
from backend.core import AffinityPredictor, FingerprintEncoder

logger = logging.getLogger(__name__)


class LoopScheduler:
    """Orchestrates the three-agent molecular discovery loop.

    Attributes:
        running: Whether the continuous loop is active.
        cognition_store: Shared knowledge store.
        researcher: Strategy proposal agent.
        engineer: Modification application agent.
        analyzer: Evaluation and recording agent.
        cycle_count: Total cycles executed.
        last_cycle_time: Timestamp of the most recent cycle.
        next_cycle_time: Scheduled time for the next cycle.
    """

    def __init__(
        self,
        cognition_store: Optional[CognitionStore] = None,
        researcher: Optional[ResearcherAgent] = None,
        engineer: Optional[EngineerAgent] = None,
        analyzer: Optional[AnalyzerAgent] = None,
        store_path: Optional[str] = None,
    ) -> None:
        """Initialize the loop scheduler.

        Creates agents and loads/creates the cognition store. If agents are
        not provided, they are constructed from default components.

        Args:
            cognition_store: Existing cognition store (creates new if None).
            researcher: Researcher agent instance (creates if None).
            engineer: Engineer agent instance (creates if None).
            analyzer: Analyzer agent instance (creates if None).
            store_path: Path for saving/loading the cognition store.
        """
        self.running = False
        self.store_path = store_path or str(settings.data_dir / "cognition_store.json")
        self.cycle_count = 0
        self.last_cycle_time: Optional[datetime] = None
        self.next_cycle_time: Optional[datetime] = None
        # Plateau convergence detector: counts cycles since the last improvement
        self._cycles_since_improvement: int = 0
        self._plateau_threshold: int = getattr(settings, "plateau_threshold", 5)
        self._last_best_affinity: float = float("inf")

        # Initialize or load cognition store
        if cognition_store is not None:
            self.cognition_store = cognition_store
        else:
            self.cognition_store = self._init_cognition_store()

        # Initialize encoder (shared dependency)
        self.encoder = FingerprintEncoder(
            radius=settings.fingerprint_radius,
            n_bits=settings.fingerprint_nbits,
        )

        # Initialize predictor (shared dependency)
        self.predictor = AffinityPredictor()

        # Initialize agents
        self.researcher = researcher or ResearcherAgent(self.cognition_store)
        self.engineer = engineer or EngineerAgent(self.encoder)
        self.analyzer = analyzer or AnalyzerAgent(
            self.predictor, self.cognition_store, self.encoder
        )

        logger.info(
            "LoopScheduler initialized (target=%s, store_path=%s)",
            self.cognition_store.target_name,
            self.store_path,
        )

    def _init_cognition_store(self) -> CognitionStore:
        """Initialize a new cognition store with default target settings.

        Returns:
            A fresh CognitionStore instance.
        """
        return CognitionStore(
            target_chembl_id=settings.target_chembl_id,
            target_name=settings.target_name,
            created_at=datetime.now(),
        )

    async def run_single_cycle(self) -> Optional[Any]:
        """Execute one full optimization cycle.

        Pipeline:
        1. Get current best from cognition store.
        2. Researcher proposes a modification strategy.
        3. Engineer applies the modification.
        4. Analyzer evaluates the candidate.
        5. Check if affinity below threshold -> trigger validation notice.
        6. Save cognition store.
        7. Return the cycle record.

        Returns:
            The CycleRecord for this cycle, or None if the cycle failed.
        """
        self.cycle_count += 1
        cycle_number = self.cycle_count
        logger.info("=" * 60)
        logger.info("Starting cycle %d", cycle_number)
        logger.info("=" * 60)

        try:
            # Step 1: Get current best
            if self.cognition_store.best_smiles_ever:
                current_best_smiles = self.cognition_store.best_smiles_ever
                current_best_fp = self.cognition_store.best_fp_ever
            else:
                # First cycle: use default parent
                current_best_smiles = getattr(settings, "default_parent_smiles", "CC(C)(C)Nc1ncnc2nc(-c3ccc(O)cc3)n(C3CC3)c12")
                current_best_fp = self.encoder.fp_to_sparse(
                    self.encoder.smiles_to_fp(current_best_smiles)
                )

            # Step 2: Researcher proposes modification
            # If we are on a plateau (no improvement for N cycles), force guided_mutation
            # so the Researcher exploits the statistical patterns instead of exploring.
            plateau_override: Optional[str] = None
            if self._cycles_since_improvement >= self._plateau_threshold:
                plateau_override = "guided_mutation"
                logger.info(
                    "PLATEAU detected (%d cycles without improvement) — "
                    "overriding strategy to guided_mutation",
                    self._cycles_since_improvement,
                )
            modification = self.researcher.propose_modification(
                current_best_smiles=current_best_smiles,
                current_best_fp=current_best_fp,
                statistical_patterns=self.cognition_store.statistical_patterns,
                accumulated_lessons=self.cognition_store.accumulated_lessons,
                cycle_number=cycle_number,
            )
            if plateau_override:
                modification["strategy"] = plateau_override

            # Step 3: Mutate SMILES with RDKit to get a real, synthesisable molecule
            # This replaces the broken fingerprint bit-flip approach.
            from backend.agents.smiles_mutator import mutate_smiles
            strategy = modification.get("strategy", "exploration")
            new_smiles_candidate, mutation_desc = mutate_smiles(
                parent_smiles=current_best_smiles,
                strategy=strategy,
                seed=cycle_number,
            )
            logger.info("SMILES mutation [%s]: %s", mutation_desc, new_smiles_candidate[:60])
            # Deduplication: if this molecule has been scored before, retry mutation
            _dedup_retries = 0
            while self.cognition_store.is_seen(new_smiles_candidate) and _dedup_retries < 5:
                _dedup_retries += 1
                logger.info(
                    "Molecule already seen (attempt %d) — retrying mutation", _dedup_retries
                )
                new_smiles_candidate, mutation_desc = mutate_smiles(
                    parent_smiles=current_best_smiles,
                    strategy=strategy,
                    seed=cycle_number + _dedup_retries * 1000,
                )
            if _dedup_retries > 0:
                logger.info(
                    "Dedup resolved after %d retries: %s", _dedup_retries, new_smiles_candidate[:60]
                )
            # Encode the new SMILES to a dense fingerprint for scoring
            _encoded = self.encoder.smiles_to_fp(new_smiles_candidate)
            # Reconstruct base fingerprint for diff computation
            base_fp_dense = np.zeros(self.encoder.n_bits, dtype=np.float32)
            base_fp_dense[list(current_best_fp)] = 1.0
            if _encoded is None:
                # Mutator returned an unparseable SMILES — fall back to parent fp
                logger.warning(
                    "smiles_to_fp returned None for mutated SMILES — using parent fp"
                )
                new_fp_dense = base_fp_dense.copy()
                new_smiles_candidate = current_best_smiles
            else:
                new_fp_dense = _encoded.astype(np.float32)
            # Attach the real SMILES and mutation description to the modification dict
            modification["new_smiles"] = new_smiles_candidate
            modification["mutation_desc"] = mutation_desc

            # Step 4: Analyzer evaluates and records
            record = self.analyzer.analyze_candidate(
                new_fp=new_fp_dense,
                parent_smiles=current_best_smiles,
                proposed_modification=modification,
                cycle_number=cycle_number,
                parent_fp=base_fp_dense,
            )

            # Step 5: Check threshold for validation trigger
            affinity_threshold_nm = getattr(settings, "affinity_threshold_nm", 10.0)
            if record.predicted_affinity_nm < affinity_threshold_nm:
                logger.info(
                    "VALIDATION TRIGGER: affinity %.3f nM below threshold %.3f nM — "
                    "would initiate experimental validation (module built separately)",
                    record.predicted_affinity_nm,
                    affinity_threshold_nm,
                )

            # Step 6: Save cognition store
            self.cognition_store.save(self.store_path)

            # Update plateau detector
            current_best = self.cognition_store.best_affinity_ever
            if current_best < self._last_best_affinity:
                self._cycles_since_improvement = 0
                self._last_best_affinity = current_best
                logger.info("Plateau counter reset (new best: %.3f nM)", current_best)
            else:
                self._cycles_since_improvement += 1
                logger.debug(
                    "No improvement this cycle (plateau count: %d/%d)",
                    self._cycles_since_improvement,
                    self._plateau_threshold,
                )

            self.last_cycle_time = datetime.now()
            logger.info(
                "Cycle %d complete. Best affinity ever: %.3f nM",
                cycle_number,
                self.cognition_store.best_affinity_ever,
            )

            # Step 7: Generate evidence PDF for new best candidates
            # Only triggered when a new best is found — one PDF per breakthrough.
            if record.is_best_so_far:
                try:
                    from backend.evidence.evidence_builder import EvidenceBuilder
                    eb = EvidenceBuilder(output_dir=str(settings.data_dir / "evidence"))
                    discovery_dict = {
                        "candidate_id": f"CAND-{cycle_number}-{int(record.predicted_affinity_nm)}",
                        "smiles": record.new_smiles if hasattr(record, 'new_smiles') else modification.get('new_smiles', current_best_smiles),
                        "predicted_affinity": record.predicted_affinity_nm,
                        "predicted_affinity_nm": record.predicted_affinity_nm,
                        "overall_pass": record.predicted_affinity_nm < affinity_threshold_nm,
                        "confidence_score": 0.678,  # model R²
                        "docking": None,
                        "admet": None,
                        "mutation_desc": modification.get('mutation_desc', 'unknown'),
                        "cycle": cycle_number,
                        "is_best_so_far": True,
                    }
                    cycle_record_dict = {
                        "cycles": [
                            {
                                "cycle": c.cycle_number if hasattr(c, 'cycle_number') else i,
                                "predicted_affinity_nm": c.predicted_affinity_nm,
                                "is_best_so_far": c.is_best_so_far,
                            }
                            for i, c in enumerate(self.cognition_store.cycles[-10:], 1)
                        ] if self.cognition_store.cycles else [{"cycle": cycle_number, "predicted_affinity_nm": record.predicted_affinity_nm, "is_best_so_far": True}],
                    }
                    target_info_dict = {
                        "chembl_id": getattr(settings, 'target_chembl_id', 'CHEMBL243'),
                        "target_name": getattr(settings, 'target_name', 'HIV-1 Protease'),
                        "uniprot_id": getattr(settings, 'target_uniprot_id', 'P04585'),
                        "organism": getattr(settings, 'target_organism', 'Human immunodeficiency virus 1'),
                        "target_type": getattr(settings, 'target_type', 'SINGLE PROTEIN'),
                    }
                    model_metrics_dict = {
                        "train_size": 4719,
                        "test_r2": 0.678,
                        "test_rmse": 0.886,
                        "model_type": "RandomForest",
                        "prediction_ci": "±1 log unit (95%)",
                    }
                    pdf_path = eb.build_evidence(
                        discovery_dict, cycle_record_dict, target_info_dict, model_metrics_dict
                    )
                    logger.info(
                        "Cycle %d: evidence PDF generated → %s",
                        cycle_number,
                        pdf_path,
                    )
                    # Attach pdf_path to record for downstream use
                    record.evidence_pdf_path = pdf_path
                except Exception as _pdf_exc:
                    logger.warning(
                        "Cycle %d: evidence PDF generation failed (non-fatal): %s",
                        cycle_number,
                        _pdf_exc,
                    )

            # Step 8: Emit best candidate to citation.is (fire-and-forget)
            # Only triggered when a new best is found to avoid spamming the API.
            if record.is_best_so_far:
                try:
                    from backend.api.candidates import emit_best_candidate_to_citation_is
                    citation_url = await emit_best_candidate_to_citation_is(
                        store=self.cognition_store,
                        ttruthdesk_url=settings.citation_is_url,
                        store_path=self.store_path,
                    )
                    if citation_url:
                        logger.info(
                            "Cycle %d: best candidate emitted to citation.is → %s",
                            cycle_number,
                            citation_url,
                        )
                except Exception as _emit_exc:
                    # Non-fatal: emission failure must never break the discovery loop
                    logger.warning(
                        "Cycle %d: citation.is emission failed (non-fatal): %s",
                        cycle_number,
                        _emit_exc,
                    )

            return record

        except Exception as exc:
            logger.error(
                "Cycle %d FAILED: %s — continuing to next cycle",
                cycle_number,
                exc,
                exc_info=True,
            )
            return None

    async def run_continuous(self) -> None:
        """Run the optimization loop continuously with fixed intervals.

        Uses asyncio.sleep between cycles. Checks self.running flag for
        graceful shutdown. Catches and logs exceptions without crashing.

        The default interval is 72 minutes (20 cycles/day).
        """
        self.running = True
        cycle_interval_seconds = getattr(settings, "cycle_interval_seconds", 4320)
        max_cycles = getattr(settings, "max_cycles", 0)
        logger.info(
            "Continuous loop started (interval=%.0f seconds, max_cycles=%s)",
            cycle_interval_seconds,
            max_cycles if max_cycles > 0 else "unlimited",
        )

        while self.running:
            self.next_cycle_time = datetime.now() + timedelta(
                seconds=cycle_interval_seconds
            )
            record = await self.run_single_cycle()

            if not self.running:
                break

            # Check max cycles
            if max_cycles > 0 and self.cycle_count >= max_cycles:
                logger.info(
                    "Reached MAX_CYCLES limit (%d), stopping.", max_cycles
                )
                self.running = False
                break

            # Sleep until next cycle
            logger.info("Sleeping for %.0f seconds until next cycle...", cycle_interval_seconds)
            await asyncio.sleep(cycle_interval_seconds)

        logger.info("Continuous loop stopped (%d total cycles).", self.cycle_count)

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the scheduler.

        Returns:
            Dict with keys: running, cycle_count, current_best_affinity,
            last_cycle_time, next_cycle_time.
        """
        return {
            "running": self.running,
            "cycle_count": self.cycle_count,
            "current_best_affinity": self.cognition_store.best_affinity_ever,
            "best_smiles": self.cognition_store.best_smiles_ever,
            "last_cycle_time": (
                self.last_cycle_time.isoformat() if self.last_cycle_time else None
            ),
            "next_cycle_time": (
                self.next_cycle_time.isoformat() if self.next_cycle_time else None
            ),
            "target": self.cognition_store.target_name,
            "target_chembl_id": self.cognition_store.target_chembl_id,
            "total_lessons": len(self.cognition_store.accumulated_lessons),
            "cycles_since_improvement": self._cycles_since_improvement,
            "plateau_threshold": self._plateau_threshold,
        }

    def start(self) -> None:
        """Start the continuous loop."""
        self.running = True
        logger.info("Scheduler start signal received.")

    def stop(self) -> None:
        """Signal the continuous loop to stop gracefully."""
        self.running = False
        logger.info("Scheduler stop signal received — will exit after current cycle.")

    def save_store(self, path: Optional[str] = None) -> None:
        """Save the cognition store to disk.

        Args:
            path: Override path (uses default store_path if None).
        """
        save_path = path or self.store_path
        self.cognition_store.save(save_path)
