"""Migrate the 44-record novus-is HIV protease seed corpus into the
asi-evolve-discovery-engine SQLite database.

Run from the repo root:
    cd /home/ubuntu/asi-evolve-discovery-engine
    python3 scripts/migrate_corpus.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database.session import create_tables, AsyncSessionLocal
from backend.database.discovery_db import create_discovery, list_discoveries

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CORPUS_PATH = Path(__file__).parent.parent / "data" / "hiv_protease_corpus.json"


async def migrate() -> None:
    corpus = json.loads(CORPUS_PATH.read_text())
    records = corpus.get("records", [])
    logger.info("Loaded %d records from corpus", len(records))

    await create_tables()

    async with AsyncSessionLocal() as session:
        # Check how many already exist
        existing = await list_discoveries(session, limit=1000)
        existing_smiles = {d.smiles for d in existing}
        logger.info("Existing DB records: %d", len(existing_smiles))

        added = 0
        skipped = 0
        for rec in records:
            smiles = rec.get("smiles", "")
            if smiles in existing_smiles:
                skipped += 1
                continue

            # Map novus-is fields to asi-evolve schema
            pic50 = rec.get("pic50", 0.0)
            # Convert pIC50 to nM: nM = 10^(9 - pIC50)
            affinity_nm = 10 ** (9 - pic50) if pic50 > 0 else None

            data = {
                "candidate_id": rec.get("name", f"seed_{added:04d}"),
                "smiles": smiles,
                "target_chembl_id": "CHEMBL2094253",
                "target_name": "HIV-1 Protease",
                "target_uniprot": "P04585",
                "predicted_affinity": pic50,
                "predicted_affinity_unit": "pIC50",
                "confidence_score": rec.get("confidence", 0.0),
                "status": "validated",  # seed records are pre-validated
            }

            try:
                await create_discovery(session, data)
                await session.commit()
                added += 1
            except Exception as exc:
                await session.rollback()
                logger.warning("Failed to insert %s: %s", rec.get("name"), exc)

        logger.info("Migration complete: %d added, %d skipped", added, skipped)


if __name__ == "__main__":
    asyncio.run(migrate())
