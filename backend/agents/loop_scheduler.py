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
from backend.core.affinity_predictor import AffinityPredictor
from backend.core.fingerprint_encoder import FingerprintEncoder
from backend.core import settings

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
        self.store_path = store_path or settings.COGNITION_STORE_PATH
        self.cycle_count = 0
        self.last_cycle_time: Optional[datetime] = None
        self.next_cycle_time: Optional[datetime] = None

        # Initialize or load cognition store
        if cognition_store is not None:
            self.cognition_store = cognition_store
        else:
            self.cognition_store = self._init_cognition_store()

        # Initialize encoder (shared dependency)
        self.encoder = FingerprintEncoder(
            fp_size=settings.FP_SIZE,
            radius=settings.FP_RADIUS,
        )

        # Initialize predictor (shared dependency)
        self.predictor = AffinityPredictor(fp_size=settings.FP_SIZE)

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
            target_chembl_id=settings.DEFAULT_TARGET_CHEMBL_ID,
            target_name=settings.DEFAULT_TARGET_NAME,
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
                current_best_smiles = settings.DEFAULT_PARENT_SMILES
                current_best_fp = self.encoder.dense_to_sparse(
                    self.encoder.encode(current_best_smiles)
                )

            # Step 2: Researcher proposes modification
            modification = self.researcher.propose_modification(
                current_best_smiles=current_best_smiles,
                current_best_fp=current_best_fp,
                statistical_patterns=self.cognition_store.statistical_patterns,
                accumulated_lessons=self.cognition_store.accumulated_lessons,
                cycle_number=cycle_number,
            )

            # Step 3: Engineer applies modification
            base_fp_dense = self.encoder.sparse_to_dense(current_best_fp)
            new_fp_dense, changed_bits = self.engineer.apply_modification(
                base_fp_dense, modification
            )

            # Step 4: Analyzer evaluates and records
            record = self.analyzer.analyze_candidate(
                new_fp=new_fp_dense,
                parent_smiles=current_best_smiles,
                proposed_modification=modification,
                cycle_number=cycle_number,
                parent_fp=base_fp_dense,
            )

            # Step 5: Check threshold for validation trigger
            if record.predicted_affinity_nm < settings.AFFINITY_THRESHOLD_NM:
                logger.info(
                    "VALIDATION TRIGGER: affinity %.3f nM below threshold %.3f nM — "
                    "would initiate experimental validation (module built separately)",
                    record.predicted_affinity_nm,
                    settings.AFFINITY_THRESHOLD_NM,
                )

            # Step 6: Save cognition store
            self.cognition_store.save(self.store_path)

            self.last_cycle_time = datetime.now()
            logger.info(
                "Cycle %d complete. Best affinity ever: %.3f nM",
                cycle_number,
                self.cognition_store.best_affinity_ever,
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
        logger.info(
            "Continuous loop started (interval=%.0f seconds)",
            settings.CYCLE_INTERVAL_SECONDS,
        )

        while self.running:
            self.next_cycle_time = datetime.now() + timedelta(
                seconds=settings.CYCLE_INTERVAL_SECONDS
            )
            record = await self.run_single_cycle()

            if not self.running:
                break

            # Check max cycles
            if settings.MAX_CYCLES > 0 and self.cycle_count >= settings.MAX_CYCLES:
                logger.info(
                    "Reached MAX_CYCLES limit (%d), stopping.", settings.MAX_CYCLES
                )
                self.running = False
                break

            # Sleep until next cycle
            sleep_seconds = settings.CYCLE_INTERVAL_SECONDS
            logger.info("Sleeping for %.0f seconds until next cycle...", sleep_seconds)
            await asyncio.sleep(sleep_seconds)

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
