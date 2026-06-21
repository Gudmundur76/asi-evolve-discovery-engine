"""Evidence generation API endpoints.

Provides REST endpoints for regenerating evidence PDFs for existing
discovery records and serving PDF files from the evidence directory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.database.discovery_db import get_discovery
from backend.database.session import get_async_session
from backend.evidence import EvidenceBuilder

logger = logging.getLogger(__name__)

router = APIRouter(tags=["evidence"])

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_evidence_builder: Optional[EvidenceBuilder] = None


def _get_evidence_builder() -> EvidenceBuilder:
    """Return a shared EvidenceBuilder singleton."""
    global _evidence_builder
    if _evidence_builder is None:
        pdf_dir = str(Path(settings.data_dir) / "pdfs")
        _evidence_builder = EvidenceBuilder(
            licensor="ASI-Evolve Discovery Engine",
            version="1.0.0",
            output_dir=pdf_dir,
        )
        logger.info("EvidenceBuilder singleton created (output_dir=%s)", pdf_dir)
    return _evidence_builder


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/generate/{discovery_id}")
async def generate_evidence_pdf(
    discovery_id: int,
    session: AsyncSession = Depends(get_async_session),
) -> Dict[str, str]:
    """Regenerate the evidence PDF for a discovery record.

    Args:
        discovery_id: Primary key of the discovery record.

    Returns:
        Dictionary with ``pdf_path`` and ``message``.

    Raises:
        HTTPException: 404 if the discovery does not exist.
    """
    try:
        record = await get_discovery(session, discovery_id)
    except Exception as exc:
        logger.error("generate_evidence_pdf query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Database query failed")

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Discovery with id={discovery_id} not found",
        )

    try:
        builder = _get_evidence_builder()

        # Build discovery dict from the ORM record
        discovery_dict = record.to_dict()

        # Provide empty/default cycle record and target info
        cycle_record_dict: Dict[str, Any] = {"cycles": []}
        target_info_dict: Dict[str, Any] = {
            "chembl_id": record.target_chembl_id or settings.target_chembl_id,
            "target_name": record.target_name or settings.target_name,
            "uniprot_id": getattr(record, "target_uniprot", None),
            "organism": "Homo sapiens",
            "target_type": "SINGLE PROTEIN",
        }
        model_metrics_dict: Dict[str, Any] = {
            "train_size": 0,
            "val_size": 0,
            "test_r2": 0.0,
            "test_rmse": 0.0,
            "model_type": getattr(settings, "model_type", "random_forest"),
            "model_version": "1.0.0",
            "prediction_ci": 0.5,
        }

        # Add a default cycle entry for the report
        cycle_record_dict["cycles"] = [
            {
                "cycle": 1,
                "type": "prediction",
                "description": f"Predicted affinity for candidate {record.candidate_id}",
                "score": record.predicted_affinity,
                "validation_passed": bool(record.overall_pass),
            }
        ]

        pdf_path = builder.build_evidence(
            discovery_dict=discovery_dict,
            cycle_record_dict=cycle_record_dict,
            target_info_dict=target_info_dict,
            model_metrics_dict=model_metrics_dict,
        )

        # Update the record with the new PDF path
        from backend.database.discovery_db import update_discovery

        await update_discovery(
            session,
            discovery_id,
            {"evidence_pdf_path": pdf_path},
        )

        logger.info("Evidence PDF regenerated for discovery %d: %s", discovery_id, pdf_path)
        return {
            "pdf_path": pdf_path,
            "message": "Evidence PDF generated successfully",
        }
    except Exception as exc:
        logger.error(
            "generate_evidence_pdf(%d) failed: %s", discovery_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to generate evidence PDF: {exc}"
        )


@router.get("/download/{filename}")
async def download_evidence_pdf(filename: str) -> Any:
    """Serve an evidence PDF file from the data/pdfs directory.

    Args:
        filename: Name of the PDF file (with or without .pdf extension).

    Returns:
        FileResponse with the PDF content.

    Raises:
        HTTPException: 404 if the file does not exist.
    """
    from fastapi.responses import FileResponse

    # Sanitize filename to prevent directory traversal
    safe_filename = os.path.basename(filename)
    if not safe_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Ensure .pdf extension
    if not safe_filename.endswith(".pdf"):
        safe_filename += ".pdf"

    pdf_dir = Path(settings.data_dir) / "pdfs"
    file_path = pdf_dir / safe_filename

    # Resolve to absolute and ensure it's within the pdf_dir
    try:
        file_path = file_path.resolve()
        pdf_dir = pdf_dir.resolve()
        if not str(file_path).startswith(str(pdf_dir)):
            raise HTTPException(status_code=400, detail="Invalid filename path")
    except (OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF file not found: {safe_filename}",
        )

    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        filename=safe_filename,
    )
