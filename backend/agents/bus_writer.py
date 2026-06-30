"""
bus_writer.py — GitHub Bus Writer for asi-evolve-discovery-engine

Publishes the best small molecule candidate from each discovery cycle
to manus-persistent-drive/bus/asi-evolve/results/.

This is the asi-evolve side of the GitHub message bus. It mirrors the
pattern used by notus-is/server/discovery/busWriter.ts, but in Python.

generic-signal-api can read these results to enrich its autonomous loop
with the best HIV protease inhibitor candidates discovered by asi-evolve.

Usage (called from hiv_loop_extension.py after each cycle):
    from backend.agents.bus_writer import write_best_candidate_to_bus
    await write_best_candidate_to_bus(candidate, cycle_stats)

Environment variables:
    BUS_REPO_PATH  — absolute path to manus-persistent-drive clone
                     (default: ../../../manus-persistent-drive relative to this file)
    GH_PAT         — GitHub PAT for git push (falls back to gh CLI auth)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_THIS_DIR = Path(__file__).parent
BUS_REPO = Path(
    os.environ.get(
        "BUS_REPO_PATH",
        str(_THIS_DIR / ".." / ".." / ".." / "manus-persistent-drive"),
    )
).resolve()
BUS_RESULTS_DIR = BUS_REPO / "bus" / "asi-evolve" / "results"
GH_PAT = os.environ.get("GH_PAT", "")

# Minimum pIC50 to publish — only publish genuinely good candidates
MIN_PIC50_TO_PUBLISH = 6.0


# ── Git helpers ───────────────────────────────────────────────────────────────

def _git(args: list[str], cwd: Path) -> bool:
    """Run a git command. Returns True on success."""
    try:
        env = os.environ.copy()
        if GH_PAT:
            # Inject PAT into credential helper for this call
            env["GIT_ASKPASS"] = "echo"
            env["GIT_USERNAME"] = "x-token"
            env["GIT_PASSWORD"] = GH_PAT
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        if result.returncode != 0:
            logger.warning("git %s failed: %s", " ".join(args), result.stderr.strip())
        return result.returncode == 0
    except Exception as exc:
        logger.warning("git %s exception: %s", " ".join(args), exc)
        return False


def _commit_and_push(result_file: Path, message: str) -> bool:
    """Stage, commit, and push a single result file to the bus repo."""
    if not BUS_REPO.exists():
        logger.warning("Bus repo not found at %s — skipping push", BUS_REPO)
        return False

    _git(["add", str(result_file.relative_to(BUS_REPO))], cwd=BUS_REPO)
    _git(["commit", "-m", message, "--allow-empty"], cwd=BUS_REPO)
    ok = _git(["push", "origin", "main"], cwd=BUS_REPO)
    if ok:
        logger.info("Bus result pushed: %s", result_file.name)
    else:
        logger.warning("Push failed for %s — result saved locally only", result_file.name)
    return ok


# ── Public API ────────────────────────────────────────────────────────────────

async def write_best_candidate_to_bus(
    candidate: Any,
    cycle_stats: Dict[str, Any],
    target: str = "HIV-1 Protease",
    chembl_id: str = "CHEMBL2094253",
) -> bool:
    """
    Write the best candidate from a discovery cycle to the GitHub bus.

    Parameters
    ----------
    candidate : ConvergenceCandidate or dict-like with .smiles, .best_pic50, etc.
    cycle_stats : dict with keys: cycle_count, day_number, candidates_generated, etc.
    target : human-readable target name
    chembl_id : ChEMBL ID of the target

    Returns True if the result was written (and pushed) successfully.
    """
    try:
        # Extract candidate fields — support both dataclass and dict
        if hasattr(candidate, "smiles"):
            smiles = candidate.smiles
            pic50 = getattr(candidate, "best_pic50", 0.0) or getattr(candidate, "mean_pic50", 0.0)
            tracks = getattr(candidate, "tracks", [])
            tanimoto = getattr(candidate, "mean_tanimoto", 0.0)
            citation_confidence = getattr(candidate, "citation_confidence", 0.0)
            pubmed_ids = getattr(candidate, "pubmed_ids", [])
            scaffold_family = getattr(candidate, "scaffold_family", "")
            rank = getattr(candidate, "rank", 1)
        else:
            smiles = candidate.get("smiles", "")
            pic50 = candidate.get("best_pic50", candidate.get("pic50", 0.0))
            tracks = candidate.get("tracks", [])
            tanimoto = candidate.get("mean_tanimoto", 0.0)
            citation_confidence = candidate.get("citation_confidence", 0.0)
            pubmed_ids = candidate.get("pubmed_ids", [])
            scaffold_family = candidate.get("scaffold_family", "")
            rank = candidate.get("rank", 1)

        if not smiles:
            logger.debug("No SMILES — skipping bus write")
            return False

        if pic50 < MIN_PIC50_TO_PUBLISH:
            logger.debug(
                "pIC50 %.2f below threshold %.2f — skipping bus write",
                pic50, MIN_PIC50_TO_PUBLISH,
            )
            return False

        # Build the result record
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y%m%dT%H%M%SZ")
        cycle = cycle_stats.get("cycle_count", 0)
        day = cycle_stats.get("day_number", 0)

        result = {
            "source": "asi-evolve",
            "publishedAt": now.isoformat(),
            "target": target,
            "chemblId": chembl_id,
            "cycle": cycle,
            "day": day,
            "bestCandidate": {
                "smiles": smiles,
                "pic50": round(pic50, 4),
                "rank": rank,
                "tracks": tracks,
                "meanTanimoto": round(tanimoto, 4),
                "citationConfidence": round(citation_confidence, 4),
                "pubmedIds": pubmed_ids,
                "scaffoldFamily": scaffold_family,
            },
            "cycleStats": {
                "candidatesGenerated": cycle_stats.get("candidates_generated", 0),
                "candidatesVerified": cycle_stats.get("candidates_verified", 0),
                "convergenceCandidates": cycle_stats.get("convergence_candidates", 0),
                "bestPic50ThisCycle": round(cycle_stats.get("best_pic50", 0.0), 4),
            },
        }

        # Write to bus
        BUS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"asi-evolve-day{day:02d}-cycle{cycle:04d}-{timestamp}.json"
        result_file = BUS_RESULTS_DIR / filename
        result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))

        logger.info(
            "Bus result written: %s (pIC50=%.2f, tracks=%s)",
            filename, pic50, tracks,
        )

        # Commit and push
        commit_msg = (
            f"feat(asi-evolve): day {day} cycle {cycle} — "
            f"pIC50={pic50:.2f} {smiles[:30]}..."
        )
        _commit_and_push(result_file, commit_msg)
        return True

    except Exception as exc:
        logger.warning("write_best_candidate_to_bus failed (non-fatal): %s", exc)
        return False


async def write_convergence_report_to_bus(
    report: Any,
    cycle_stats: Dict[str, Any],
    target: str = "HIV-1 Protease",
    chembl_id: str = "CHEMBL2094253",
) -> int:
    """
    Write all convergence candidates from a milestone report to the bus.
    Returns the number of candidates published.
    """
    if not report or not getattr(report, "candidates", None):
        return 0

    published = 0
    for candidate in report.candidates:
        ok = await write_best_candidate_to_bus(
            candidate, cycle_stats, target=target, chembl_id=chembl_id
        )
        if ok:
            published += 1

    if published:
        logger.info(
            "Published %d convergence candidates to bus (day %d, milestone %s)",
            published,
            getattr(report, "day", 0),
            getattr(report, "milestone", ""),
        )
    return published
