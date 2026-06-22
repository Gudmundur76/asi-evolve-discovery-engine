"""End-to-end integration test for the asi-evolve-discovery-engine HIV protease build.

Tests all new components added to the Kimi base:
1. Config — HIV protease target settings
2. SQLite database — all tables, CRUD (async via asyncio.run)
3. MultiTrackEngineer — all 4 tracks generate valid SMILES
4. CitationGate — live call to citation.manus.space
5. ConvergenceDetector — Tanimoto analysis on seed corpus
6. persist_to_drive — dry run (no actual push)

Run from the repo root:
    python3 scripts/test_integration.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add backend to path — must be before any backend imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("integration_test")

PASS = "✓ PASS"
FAIL = "✗ FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    icon = "✓" if condition else "✗"
    logger.info("%s %s %s", icon, name, f"— {detail}" if detail else "")
    return condition


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Config
# ─────────────────────────────────────────────────────────────────────────────
def test_config() -> None:
    logger.info("\n=== Phase 1: Config ===")
    try:
        from backend.config import Settings
        s = Settings()
        check("Config: target_chembl_id is HIV protease", s.target_chembl_id == "CHEMBL2094253",
              f"got {s.target_chembl_id}")
        check("Config: target_name contains HIV", "HIV" in s.target_name,
              f"got {s.target_name}")
        check("Config: citation_is_url is citation.manus.space",
              "citation.manus.space" in s.citation_is_url,
              f"got {s.citation_is_url}")
        check("Config: citation_confidence_threshold is 0.85",
              s.citation_confidence_threshold == 0.85,
              f"got {s.citation_confidence_threshold}")
        check("Config: convergence_tanimoto_threshold is 0.70",
              s.convergence_tanimoto_threshold == 0.70,
              f"got {s.convergence_tanimoto_threshold}")
        check("Config: target_uniprot is P04585",
              s.target_uniprot == "P04585",
              f"got {s.target_uniprot}")
    except Exception as e:
        check("Config: import and instantiate", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Database (async)
# Table names in models.py: "discovery", "cognition_log" (no model_versions table)
# Discovery columns: candidate_id, smiles, target_chembl_id, predicted_affinity, etc.
# ─────────────────────────────────────────────────────────────────────────────
async def _test_database_async() -> None:
    from backend.database.session import engine, create_tables
    from backend.database.models import Discovery
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import select, func, inspect as sa_inspect

    # Create tables
    await create_tables()
    check("DB: create_tables() succeeded", True)

    # Check tables exist via reflection
    async with engine.connect() as conn:
        tables = await conn.run_sync(
            lambda sync_conn: sa_inspect(sync_conn).get_table_names()
        )
    check("DB: discovery table exists", "discovery" in tables, str(tables))
    check("DB: cognition_log table exists", "cognition_log" in tables, str(tables))

    # Check seed corpus count
    async with AsyncSession(engine) as session:
        count_result = await session.execute(select(func.count()).select_from(Discovery))
        count = count_result.scalar()
        check("DB: seed corpus migrated (≥39 records)", count >= 39, f"got {count}")

        # Check Darunavir by candidate_id (not compound_name — that column does not exist)
        result = await session.execute(
            select(Discovery).where(Discovery.candidate_id.like("%darunavir%"))
        )
        darunavir = result.scalars().first()
        check("DB: Darunavir seed record present", darunavir is not None,
              "not found" if darunavir is None else f"id={darunavir.id}, candidate_id={darunavir.candidate_id}")

        # Check a record has SMILES
        first = await session.execute(select(Discovery).limit(1))
        first_rec = first.scalars().first()
        check("DB: first record has SMILES", first_rec is not None and bool(first_rec.smiles),
              f"smiles={first_rec.smiles[:30] if first_rec else 'None'}...")


def test_database() -> None:
    logger.info("\n=== Phase 2: Database ===")
    try:
        asyncio.run(_test_database_async())
    except Exception as e:
        check("DB: async test", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: MultiTrackEngineer
# __init__ takes: fingerprint_radius, fingerprint_nbits, novelty_tanimoto_threshold
# generate_track takes: track (str), n_candidates (int), cognition_patterns (dict)
# Returns: List[TrackCandidate] — each has .smiles, .track attributes
# ─────────────────────────────────────────────────────────────────────────────
def test_multi_track_engineer() -> None:
    logger.info("\n=== Phase 3: MultiTrackEngineer ===")
    try:
        from backend.agents.multi_track_engineer import MultiTrackEngineer, TRACK_A, TRACK_B, TRACK_C, TRACK_D
        from rdkit import Chem

        engineer = MultiTrackEngineer(
            fingerprint_radius=2,
            fingerprint_nbits=2048,
        )
        check("MultiTrackEngineer: instantiates", True)

        for track_id in [TRACK_A, TRACK_B, TRACK_C, TRACK_D]:
            candidates = engineer.generate_track(track=track_id, n_candidates=5)
            valid_smiles = [
                c for c in candidates
                if Chem.MolFromSmiles(c.smiles) is not None
            ]
            check(
                f"Track {track_id}: generates 5 candidates",
                len(candidates) == 5,
                f"got {len(candidates)}",
            )
            check(
                f"Track {track_id}: all SMILES valid",
                len(valid_smiles) == len(candidates),
                f"{len(valid_smiles)}/{len(candidates)} valid",
            )
            check(
                f"Track {track_id}: candidates have track field",
                all(c.track == track_id for c in candidates),
                f"tracks={[c.track for c in candidates[:2]]}",
            )

    except Exception as e:
        check("MultiTrackEngineer: import and run", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: CitationGate (live call)
# ─────────────────────────────────────────────────────────────────────────────
def test_citation_gate() -> None:
    logger.info("\n=== Phase 4: CitationGate ===")
    try:
        from backend.agents.citation_gate import CitationGate

        gate = CitationGate(confidence_threshold=0.85)
        check("CitationGate: instantiates", True)

        # Darunavir — should pass
        result = gate.verify(
            smiles="CC(C)(C)OC(=O)N[C@@H](Cc1ccccc1)[C@@H](O)CN1C[C@@H]2CCOC[C@@H]2C1=O",
            compound_name="Darunavir",
            predicted_pic50=9.1,
        )
        check("CitationGate: Darunavir HTTP success", result.error is None,
              result.error or "")
        check("CitationGate: Darunavir verdict Supported", result.verdict == "Supported",
              f"got {result.verdict}")
        check("CitationGate: Darunavir confidence ≥ 0.85", result.confidence_score >= 0.85,
              f"got {result.confidence_score:.3f}")
        check("CitationGate: Darunavir gate_passed=True", result.gate_passed,
              f"gate_passed={result.gate_passed}")
        check("CitationGate: Darunavir has PubMed IDs", len(result.pubmed_ids) > 0,
              f"got {len(result.pubmed_ids)} PMIDs")
        check("CitationGate: latency < 10s", result.latency_ms < 10000,
              f"{result.latency_ms:.0f}ms")

        # A clearly non-drug random SMILES — should return low confidence or insufficient evidence
        # Note: citation.manus.space is a general knowledge graph; it may still return "Supported"
        # for some claims. We test that the gate correctly handles the response, not that it
        # always rejects non-drugs (the system is not a drug classifier).
        result_low = gate.verify(
            smiles="CCCCCCCCCCCCCCCCCC",  # octadecane — no biological activity
            compound_name="Octadecane",
            predicted_pic50=1.5,
        )
        check("CitationGate: Octadecane call succeeds (no error)", result_low.error is None,
              result_low.error or "")
        check("CitationGate: Octadecane gate_passed reflects confidence threshold",
              result_low.gate_passed == (result_low.confidence_score >= 0.85),
              f"gate_passed={result_low.gate_passed}, conf={result_low.confidence_score:.3f}")

    except Exception as e:
        check("CitationGate: import and run", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: ConvergenceDetector
# ─────────────────────────────────────────────────────────────────────────────
def test_convergence_detector() -> None:
    logger.info("\n=== Phase 5: ConvergenceDetector ===")
    try:
        from backend.agents.convergence_detector import ConvergenceDetector

        detector = ConvergenceDetector(data_dir="/tmp/convergence_test")
        check("ConvergenceDetector: instantiates", True)

        check("ConvergenceDetector: should_run day 7", detector.should_run(7))
        check("ConvergenceDetector: should_run day 15", detector.should_run(15))
        check("ConvergenceDetector: should_run day 30", detector.should_run(30))
        check("ConvergenceDetector: should_run day 3 is False", not detector.should_run(3))

        check("ConvergenceDetector: threshold day 7 = 0.70",
              detector.get_threshold(7) == 0.70)
        check("ConvergenceDetector: threshold day 15 = 0.75",
              detector.get_threshold(15) == 0.75)
        check("ConvergenceDetector: threshold day 30 = 0.80",
              detector.get_threshold(30) == 0.80)

        # Synthetic corpus — Darunavir in Track A and Track B should converge
        corpus = [
            {
                "smiles": "CC(C)(C)OC(=O)N[C@@H](Cc1ccccc1)[C@@H](O)CN1C[C@@H]2CCOC[C@@H]2C1=O",
                "track": "A",
                "predicted_pic50": 9.1,
                "citation_confidence": 0.99,
                "pubmed_ids": ["40446126"],
                "scaffold_family": "hydroxyethylamine",
            },
            {
                "smiles": "CC(C)(C)OC(=O)N[C@@H](Cc1ccccc1)[C@@H](O)CN1C[C@@H]2CCOC[C@@H]2C1=O",
                "track": "B",
                "predicted_pic50": 9.0,
                "citation_confidence": 0.98,
                "pubmed_ids": ["39697065"],
                "scaffold_family": "hydroxyethylamine",
            },
            {
                "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
                "track": "C",
                "predicted_pic50": 5.5,
                "citation_confidence": 0.40,
                "pubmed_ids": [],
                "scaffold_family": "ibuprofen",
            },
        ]

        report = detector.analyse(corpus, day=7)
        check("ConvergenceDetector: report returned", report is not None)
        check("ConvergenceDetector: day 7 threshold = 0.70", report.threshold == 0.70)
        check("ConvergenceDetector: found ≥1 convergence candidate",
              len(report.candidates) >= 1,
              f"found {len(report.candidates)}")
        if report.candidates:
            c = report.candidates[0]
            check("ConvergenceDetector: candidate has tracks A and B",
                  "A" in c.tracks and "B" in c.tracks,
                  f"tracks={c.tracks}")
            check("ConvergenceDetector: candidate pIC50 ≥ 9.0",
                  c.best_pic50 >= 9.0,
                  f"best_pic50={c.best_pic50}")

    except Exception as e:
        check("ConvergenceDetector: import and run", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: persist_to_drive (dry run)
# ─────────────────────────────────────────────────────────────────────────────
def test_persist_script() -> None:
    logger.info("\n=== Phase 6: persist_to_drive (dry run) ===")
    try:
        script = Path(__file__).parent / "persist_to_drive.py"
        check("persist_to_drive.py exists", script.exists(), str(script))

        import importlib.util
        spec = importlib.util.spec_from_file_location("persist_to_drive", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        check("persist_to_drive: imports cleanly", True)

        entry = mod.build_daily_entry(
            day=4,
            cycles=5,
            summary="Integration test dry run",
            run_data={
                "corpus_size": 44,
                "candidates_generated": 150,
                "candidates_verified": 2,
                "best_pic50": 9.11,
                "convergence_candidates": 0,
                "citation_pass_rate": "12/150",
            },
        )
        check("persist_to_drive: build_daily_entry returns string", isinstance(entry, str))
        check("persist_to_drive: entry contains day 04", "Day 04" in entry, entry[:100])

    except Exception as e:
        check("persist_to_drive: dry run", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
def print_summary() -> int:
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    total = len(results)

    print("\n" + "=" * 70)
    print(f"INTEGRATION TEST SUMMARY: {passed}/{total} passed")
    print("=" * 70)
    for name, status, detail in results:
        line = f"  {status}  {name}"
        if detail and status == FAIL:
            line += f"\n         → {detail}"
        print(line)
    print("=" * 70)

    if failed > 0:
        print(f"\n{failed} test(s) FAILED")
        return 1
    else:
        print("\nAll tests PASSED")
        return 0


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)

    test_config()
    test_database()
    test_multi_track_engineer()
    test_citation_gate()
    test_convergence_detector()
    test_persist_script()

    sys.exit(print_summary())
