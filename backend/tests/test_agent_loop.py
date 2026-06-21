"""Test script for the ASI-Evolve three-agent loop.

Runs 3 manual cycles through the full loop (Researcher -> Engineer -> Analyzer)
and verifies:
- The cognition store accumulates cycles correctly
- Statistical patterns are updated
- The best affinity improves (or at least changes)
- Save/load round-trip works
"""

import asyncio
import logging
import os
import sys
import tempfile

# Ensure the project root is on the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.agents.analyzer import AnalyzerAgent
from backend.agents.cognition_store import CognitionStore
from backend.agents.engineer import EngineerAgent
from backend.agents.loop_scheduler import LoopScheduler
from backend.agents.researcher import ResearcherAgent
from backend.core.affinity_predictor import AffinityPredictor
from backend.core.fingerprint_encoder import FingerprintEncoder
from backend.core import settings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format=settings.LOG_FORMAT,
)
logger = logging.getLogger("test_agent_loop")


async def run_test_cycles(n_cycles: int = 3) -> None:
    """Run N manual cycles through the agent loop and verify behavior.

    Args:
        n_cycles: Number of cycles to execute.
    """
    logger.info("=" * 70)
    logger.info("ASI-Evolve Agent Loop Test — %d manual cycles", n_cycles)
    logger.info("=" * 70)

    # Create temporary store path
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        store_path = tmp.name

    try:
        # Initialize shared components
        encoder = FingerprintEncoder(fp_size=settings.FP_SIZE)
        predictor = AffinityPredictor(fp_size=settings.FP_SIZE)

        # Initial parent SMILES (gefitinib-like)
        parent_smiles = settings.DEFAULT_PARENT_SMILES
        logger.info("Parent SMILES: %s", parent_smiles[:60])

        # Encode parent
        parent_fp = encoder.encode(parent_smiles)
        parent_fp_sparse = encoder.dense_to_sparse(parent_fp)
        logger.info("Parent fingerprint: %d bits set", len(parent_fp_sparse))

        # Create cognition store
        store = CognitionStore(
            target_chembl_id=settings.DEFAULT_TARGET_CHEMBL_ID,
            target_name=settings.DEFAULT_TARGET_NAME,
            created_at=__import__("datetime").datetime.now(),
            best_affinity_ever=float("inf"),
        )

        # Create agents
        researcher = ResearcherAgent(store)
        engineer = EngineerAgent(encoder)
        analyzer = AnalyzerAgent(predictor, store, encoder)

        # Create scheduler (for status tracking, but we drive manually)
        scheduler = LoopScheduler(
            cognition_store=store,
            researcher=researcher,
            engineer=engineer,
            analyzer=analyzer,
            store_path=store_path,
        )

        logger.info("\n--- Initial state ---")
        logger.info("Best affinity: %.3f nM", store.best_affinity_ever)
        logger.info("Cycles: %d", len(store.cycles))

        # Run cycles
        for cycle_num in range(1, n_cycles + 1):
            logger.info("\n" + "=" * 70)
            logger.info("CYCLE %d", cycle_num)
            logger.info("=" * 70)

            # Get current best
            if store.best_smiles_ever:
                current_smiles = store.best_smiles_ever
                current_fp = store.best_fp_ever
            else:
                current_smiles = parent_smiles
                current_fp = parent_fp_sparse

            # 1. Researcher proposes modification
            logger.info("\n[1] RESEARCHER: Proposing modification...")
            modification = researcher.propose_modification(
                current_best_smiles=current_smiles,
                current_best_fp=current_fp,
                statistical_patterns=store.statistical_patterns,
                accumulated_lessons=store.accumulated_lessons,
                cycle_number=cycle_num,
            )
            logger.info("    Strategy: %s", modification["strategy"])
            logger.info(
                "    Target bits: %s",
                modification["target_bits"][:10],
            )
            logger.info("    Rationale: %s", modification["rationale"][:120])
            logger.info("    Confidence: %.3f", modification["confidence"])

            # 2. Engineer applies modification
            logger.info("\n[2] ENGINEER: Applying modification...")
            base_fp_dense = encoder.sparse_to_dense(current_fp)
            new_fp_dense, changed_bits = engineer.apply_modification(
                base_fp_dense, modification
            )
            new_fp_sparse = encoder.dense_to_sparse(new_fp_dense)
            logger.info("    Changed %d bits", len(changed_bits))
            logger.info(
                "    New fingerprint: %d bits set (was %d)",
                len(new_fp_sparse),
                len(current_fp),
            )

            # 3. Analyzer evaluates candidate
            logger.info("\n[3] ANALYZER: Evaluating candidate...")
            record = analyzer.analyze_candidate(
                new_fp=new_fp_dense,
                parent_smiles=current_smiles,
                proposed_modification=modification,
                cycle_number=cycle_num,
                parent_fp=base_fp_dense,
            )
            logger.info("    Predicted affinity: %.3f nM", record.predicted_affinity_nm)
            logger.info("    Improvement: %+.3f nM", record.improvement)
            logger.info("    Best so far: %s", record.is_best_so_far)
            logger.info("    Lesson: %s", record.lesson[:150])

            # Check threshold
            if record.predicted_affinity_nm < settings.AFFINITY_THRESHOLD_NM:
                logger.info(
                    "    *** VALIDATION WORTHY: %.3f nM < %.3f nM threshold ***",
                    record.predicted_affinity_nm,
                    settings.AFFINITY_THRESHOLD_NM,
                )

            scheduler.cycle_count = cycle_num
            scheduler.last_cycle_time = __import__("datetime").datetime.now()

        # Post-cycle verification
        logger.info("\n" + "=" * 70)
        logger.info("VERIFICATION")
        logger.info("=" * 70)

        # Verify cycles were recorded
        assert len(store.cycles) == n_cycles, (
            f"Expected {n_cycles} cycles, got {len(store.cycles)}"
        )
        logger.info("[PASS] All %d cycles recorded", n_cycles)

        # Verify lessons accumulated
        assert len(store.accumulated_lessons) == n_cycles, (
            f"Expected {n_cycles} lessons, got {len(store.accumulated_lessons)}"
        )
        logger.info("[PASS] All %d lessons accumulated", n_cycles)

        # Verify statistics were updated
        assert len(store.statistical_patterns) > 0, "No statistical patterns recorded"
        logger.info(
            "[PASS] %d bits have statistical patterns",
            len(store.statistical_patterns),
        )

        # Verify best affinity was tracked
        assert store.best_affinity_ever < float("inf"), "Best affinity never updated"
        logger.info(
            "[PASS] Best affinity tracked: %.3f nM",
            store.best_affinity_ever,
        )

        # Verify top candidates
        top5 = store.get_top_candidates(n=5)
        assert len(top5) <= 5, "Too many top candidates returned"
        assert all(
            top5[i].predicted_affinity_nm <= top5[i + 1].predicted_affinity_nm
            for i in range(len(top5) - 1)
        ), "Top candidates not sorted by affinity"
        logger.info("[PASS] Top candidates sorted correctly")

        # Verify bit statistics are sorted
        bit_stats = store.get_bit_statistics()
        values = [s.get("avg_improvement", 0.0) for s in bit_stats.values()]
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1)), (
            "Bit statistics not sorted by avg_improvement"
        )
        logger.info("[PASS] Bit statistics sorted correctly")

        # Test save
        logger.info("\n--- Testing save/load ---")
        store.save(store_path)
        assert os.path.exists(store_path), "Store file not created"
        logger.info("[PASS] Store saved to %s", store_path)

        # Test load
        loaded = CognitionStore.load(store_path)
        assert len(loaded.cycles) == len(store.cycles), "Loaded cycle count mismatch"
        assert (
            loaded.best_affinity_ever == store.best_affinity_ever
        ), "Loaded best affinity mismatch"
        assert len(loaded.statistical_patterns) == len(
            store.statistical_patterns
        ), "Loaded patterns mismatch"
        logger.info("[PASS] Store loaded correctly")

        # Test lessons summary
        summary = store.get_lessons_summary(n_recent=3)
        assert len(summary) > 0, "Lessons summary is empty"
        assert "COGNITION STORE LESSONS SUMMARY" in summary
        logger.info("[PASS] Lessons summary generated")

        # Print status
        status = scheduler.get_status()
        logger.info("\n--- Scheduler Status ---")
        for key, value in status.items():
            logger.info("  %s: %s", key, value)

        # Print lessons summary excerpt
        logger.info("\n--- Lessons Summary (excerpt) ---")
        for line in summary.split("\n")[:20]:
            logger.info("  %s", line)

        logger.info("\n" + "=" * 70)
        logger.info("ALL TESTS PASSED")
        logger.info("=" * 70)

    finally:
        # Cleanup
        if os.path.exists(store_path):
            os.unlink(store_path)
            logger.info("Cleaned up temp store: %s", store_path)


if __name__ == "__main__":
    asyncio.run(run_test_cycles(n_cycles=3))
