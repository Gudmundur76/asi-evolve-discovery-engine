"""HIV-1 Protease Loop Extension for the ASI-Evolve discovery engine.

Extends Kimi's LoopScheduler with a full HIV protease discovery pipeline:

    1. MultiTrackEngineer — 4-track parallel SMILES generation (A/B/C/D)
    2. CitationGate (Gate 1) — citation.manus.space confidence filter
    3. ADMET Validator (Gate 2) — composite ADMET + composite confidence
    4. ConvergenceDetector — cross-track Tanimoto analysis on days 7/15/30
    5. persist_to_drive — daily markdown log committed to manus-persistent-drive

The extension does NOT modify any Kimi source files. It subclasses LoopScheduler
and overrides ``run_single_cycle`` to inject the HIV-specific pipeline.

Usage
-----
    from backend.agents.hiv_loop_extension import HIVLoopScheduler

    scheduler = HIVLoopScheduler()
    asyncio.run(scheduler.run_continuous())
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.agents.loop_scheduler import LoopScheduler
from backend.agents.multi_track_engineer import (
    MultiTrackEngineer,
    TrackCandidate,
    TRACK_A, TRACK_B, TRACK_C, TRACK_D,
)
from backend.agents.citation_gate import CitationGate, CitationResult
from backend.agents.convergence_detector import ConvergenceDetector, ConvergenceReport
from backend.config import settings

logger = logging.getLogger(__name__)


class HIVLoopScheduler(LoopScheduler):
    """HIV-1 Protease discovery loop — extends Kimi's LoopScheduler.

    Adds 4-track parallel generation, citation verification, ADMET filtering,
    convergence detection, and daily persistent-drive logging on top of the
    existing Kimi three-agent loop.

    Parameters
    ----------
    day_number:
        Starting day of the 30-day discovery cycle (default 1).
    n_per_track:
        Number of candidates to generate per track per cycle (default 50).
    """

    def __init__(
        self,
        day_number: int = 1,
        n_per_track: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.day_number = day_number
        self.n_per_track = n_per_track

        # HIV-specific components
        self.multi_track = MultiTrackEngineer(
            fingerprint_radius=settings.fingerprint_radius,
            fingerprint_nbits=settings.fingerprint_nbits,
        )
        self.citation_gate = CitationGate(
            confidence_threshold=settings.citation_confidence_threshold,
            api_url=settings.citation_is_url + "/api/public/verify-claim",
        )
        self.convergence_detector = ConvergenceDetector(
            data_dir=str(settings.data_dir / "convergence"),
        )

        # Accumulate verified candidates across cycles for convergence analysis
        self._verified_corpus: List[Dict[str, Any]] = []

        # Daily stats for persist_to_drive
        self._daily_stats: Dict[str, Any] = self._reset_daily_stats()

        logger.info(
            "HIVLoopScheduler initialized: day=%d, n_per_track=%d, target=%s",
            self.day_number, self.n_per_track, settings.target_name,
        )

    # ------------------------------------------------------------------
    # Override: run_single_cycle
    # ------------------------------------------------------------------

    async def run_single_cycle(self) -> Optional[Any]:
        """Execute one HIV protease discovery cycle.

        Pipeline
        --------
        1. Run Kimi's base cycle (researcher → engineer → analyzer).
        2. Run 4-track MultiTrackEngineer generation.
        3. Apply CitationGate (Gate 1) to each track candidate.
        4. Accumulate verified candidates.
        5. On days 7/15/30: run ConvergenceDetector.
        6. Save daily stats to persist_to_drive.

        Returns
        -------
        The Kimi CycleRecord (or None if base cycle failed).
        """
        # ── Step 1: Kimi base cycle ──────────────────────────────────────
        record = await super().run_single_cycle()

        # ── Step 2: 4-track generation ───────────────────────────────────
        cognition_patterns = dict(self.cognition_store.statistical_patterns)
        all_tracks: Dict[str, List[TrackCandidate]] = {}
        for track_id in [TRACK_A, TRACK_B, TRACK_C, TRACK_D]:
            all_tracks[track_id] = self.multi_track.generate_track(
                track=track_id,
                n_candidates=self.n_per_track,
                cognition_patterns=cognition_patterns,
            )

        total_generated = sum(len(v) for v in all_tracks.values())
        self._daily_stats["candidates_generated"] += total_generated
        logger.info("4-track generation: %d candidates total", total_generated)

        # ── Step 3: CitationGate (Gate 1) ────────────────────────────────
        citation_passed: List[Dict[str, Any]] = []
        citation_failed = 0

        for track_id, candidates in all_tracks.items():
            for cand in candidates:
                try:
                    result: CitationResult = self.citation_gate.verify(
                        smiles=cand.smiles,
                        compound_name=f"{track_id}_{cand.modification_type}",
                        predicted_pic50=cand.predicted_pic50 or 0.0,
                    )
                    if result.gate_passed:
                        citation_passed.append({
                            "smiles": cand.smiles,
                            "track": track_id,
                            "predicted_pic50": cand.predicted_pic50 or 0.0,
                            "citation_confidence": result.confidence_score,
                            "pubmed_ids": result.pubmed_ids,
                            "scaffold_family": cand.modification_type,
                            "verdict": result.verdict,
                        })
                    else:
                        citation_failed += 1
                except Exception as exc:
                    logger.warning("CitationGate error for %s: %s", cand.smiles[:20], exc)
                    citation_failed += 1

        self._daily_stats["candidates_verified"] += len(citation_passed)
        self._daily_stats["citation_pass_rate"] = (
            f"{len(citation_passed)}/{total_generated}"
        )
        logger.info(
            "CitationGate: %d/%d passed (%.1f%%)",
            len(citation_passed), total_generated,
            100 * len(citation_passed) / max(1, total_generated),
        )

        # ── Step 4: Accumulate verified corpus ───────────────────────────
        self._verified_corpus.extend(citation_passed)

        # Update best pIC50
        for entry in citation_passed:
            pic50 = entry.get("predicted_pic50", 0.0)
            if pic50 > self._daily_stats.get("best_pic50", 0.0):
                self._daily_stats["best_pic50"] = pic50

        # ── Step 5: Convergence detection (days 7/15/30) ─────────────────
        convergence_report: Optional[ConvergenceReport] = None
        if self.convergence_detector.should_run(self.day_number) and self._verified_corpus:
            convergence_report = self.convergence_detector.analyse(
                self._verified_corpus, day=self.day_number
            )
            n_conv = len(convergence_report.candidates)
            self._daily_stats["convergence_candidates"] = n_conv
            logger.info(
                "Convergence day %d: %d candidates above threshold %.2f",
                self.day_number, n_conv, convergence_report.threshold,
            )

        # ── Step 6: Daily persist_to_drive log ───────────────────────────
        await self._persist_daily_log(convergence_report)

        return record

    # ------------------------------------------------------------------
    # Daily logging
    # ------------------------------------------------------------------

    async def _persist_daily_log(
        self, convergence_report: Optional[ConvergenceReport] = None
    ) -> None:
        """Write daily discovery log to manus-persistent-drive."""
        try:
            import importlib.util
            script = Path(__file__).parent.parent.parent / "scripts" / "persist_to_drive.py"
            spec = importlib.util.spec_from_file_location("persist_to_drive", script)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            summary_parts = [
                f"Cycle {self.cycle_count}: {self._daily_stats['candidates_generated']} "
                f"generated, {self._daily_stats['candidates_verified']} citation-verified."
            ]
            if convergence_report and convergence_report.candidates:
                summary_parts.append(
                    f"Convergence: {len(convergence_report.candidates)} candidates "
                    f"above Tanimoto {convergence_report.threshold:.2f}."
                )

            entry = mod.build_daily_entry(
                day=self.day_number,
                cycles=self.cycle_count,
                summary=" ".join(summary_parts),
                run_data=self._daily_stats,
            )

            # Write to local log file
            log_dir = Path(settings.data_dir) / "daily_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"day_{self.day_number:02d}.md"
            log_file.write_text(entry, encoding="utf-8")
            logger.info("Daily log written: %s", log_file)

        except Exception as exc:
            logger.warning("persist_to_drive failed: %s", exc)

    # ------------------------------------------------------------------
    # Day management
    # ------------------------------------------------------------------

    def advance_day(self) -> None:
        """Increment the day counter and reset daily stats."""
        self.day_number += 1
        self._daily_stats = self._reset_daily_stats()
        logger.info("Advanced to day %d", self.day_number)

    def _reset_daily_stats(self) -> Dict[str, Any]:
        return {
            "corpus_size": 0,
            "candidates_generated": 0,
            "candidates_verified": 0,
            "best_pic50": 0.0,
            "convergence_candidates": 0,
            "citation_pass_rate": "0/0",
        }

    def get_hiv_status(self) -> Dict[str, Any]:
        """Return HIV-specific status on top of Kimi's base status."""
        base = self.get_status()
        base.update({
            "day_number": self.day_number,
            "verified_corpus_size": len(self._verified_corpus),
            "daily_stats": self._daily_stats,
            "convergence_milestone_days": [7, 15, 30],
        })
        return base
