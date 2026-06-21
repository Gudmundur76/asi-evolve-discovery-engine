"""
SQLAlchemy ORM models for the molecular discovery engine.

Implements two core tables exactly as specified in SPEC section 4.13:

* :class:`Discovery` -- full discovery record with all scores and metadata
* :class:`CognitionLog` -- audit log of every cognition cycle

All models use SQLAlchemy 2.0 style with :func:`mapped_column` and
:type-hinted :class:`Mapped` attributes.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""

    pass


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StatusEnum(str, enum.Enum):
    """Lifecycle status of a discovery record."""

    PENDING = "pending"
    VALIDATED = "validated"
    FAILED = "failed"
    ARCHIVED = "archived"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class Discovery(Base):
    """A single validated molecular discovery.

    This table stores the full output of the validation pipeline for each
    candidate compound that reaches the validation stage.
    """

    __tablename__ = "discovery"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Candidate identifiers
    candidate_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    smiles: Mapped[str] = mapped_column(Text, nullable=False)
    inchi_key: Mapped[Optional[str]] = mapped_column(String(27), nullable=True, index=True)

    # Target information
    target_chembl_id: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True, index=True
    )
    target_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    target_uniprot: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)

    # Predictive scoring (from ML model)
    predicted_affinity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    predicted_affinity_unit: Mapped[str] = mapped_column(
        String(8), nullable=False, default="pIC50"
    )
    confidence_score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )

    # Docking results
    docking_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    docking_pass: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    docking_poses_path: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
    docking_best_mode: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    docking_best_rmsd_lb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    docking_best_rmsd_ub: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ADMET profile (stored as denormalised columns for fast queries)
    admet_mw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    admet_logp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    admet_hbd: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    admet_hba: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    admet_tpsa: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    admet_rotatable_bonds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    admet_lipinski_violations: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    admet_synthetic_accessibility: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    admet_gi_absorption: Mapped[Optional[str]] = mapped_column(
        String(8), nullable=True
    )
    admet_bbb_permeable: Mapped[Optional[bool]] = mapped_column(
        Boolean, nullable=True
    )
    admet_pgp_substrate: Mapped[Optional[bool]] = mapped_column(
        Boolean, nullable=True
    )
    admet_druglikeness_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    admet_medicinal_chemistry_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    admet_is_druglike: Mapped[Optional[bool]] = mapped_column(
        Boolean, nullable=True
    )
    admet_overall_pass: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    admet_raw: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, doc="Full raw ADMET dict from SwissADME client"
    )

    # Validation outcome
    overall_pass: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    status: Mapped[StatusEnum] = mapped_column(
        String(16),
        nullable=False,
        default=StatusEnum.PENDING,
    )

    # Evidence
    evidence_pdf_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    fingerprint_hex: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    modification_history: Mapped[Optional[list]] = mapped_column(
        JSON, nullable=True, default=list
    )

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    cognition_logs: Mapped[List["CognitionLog"]] = relationship(
        "CognitionLog",
        back_populates="discovery",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<Discovery(id={self.id}, candidate_id={self.candidate_id!r}, "
            f"smiles={self.smiles[:30]!r}..., overall_pass={self.overall_pass})>"
        )

    def to_dict(self) -> dict:
        """Serialise the discovery record to a plain dictionary."""
        return {
            "id": self.id,
            "candidate_id": self.candidate_id,
            "smiles": self.smiles,
            "inchi_key": self.inchi_key,
            "target_chembl_id": self.target_chembl_id,
            "target_name": self.target_name,
            "target_uniprot": self.target_uniprot,
            "predicted_affinity": self.predicted_affinity,
            "predicted_affinity_unit": self.predicted_affinity_unit,
            "confidence_score": self.confidence_score,
            "docking_score": self.docking_score,
            "docking_pass": self.docking_pass,
            "docking_poses_path": self.docking_poses_path,
            "docking_best_mode": self.docking_best_mode,
            "docking_best_rmsd_lb": self.docking_best_rmsd_lb,
            "docking_best_rmsd_ub": self.docking_best_rmsd_ub,
            "admet_mw": self.admet_mw,
            "admet_logp": self.admet_logp,
            "admet_hbd": self.admet_hbd,
            "admet_hba": self.admet_hba,
            "admet_tpsa": self.admet_tpsa,
            "admet_rotatable_bonds": self.admet_rotatable_bonds,
            "admet_lipinski_violations": self.admet_lipinski_violations,
            "admet_synthetic_accessibility": self.admet_synthetic_accessibility,
            "admet_gi_absorption": self.admet_gi_absorption,
            "admet_bbb_permeable": self.admet_bbb_permeable,
            "admet_pgp_substrate": self.admet_pgp_substrate,
            "admet_druglikeness_score": self.admet_druglikeness_score,
            "admet_medicinal_chemistry_score": self.admet_medicinal_chemistry_score,
            "admet_is_druglike": self.admet_is_druglike,
            "admet_overall_pass": self.admet_overall_pass,
            "admet_raw": self.admet_raw,
            "overall_pass": self.overall_pass,
            "status": self.status.value if isinstance(self.status, StatusEnum) else self.status,
            "evidence_pdf_path": self.evidence_pdf_path,
            "fingerprint_hex": self.fingerprint_hex,
            "modification_history": self.modification_history,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ---------------------------------------------------------------------------
# CognitionLog
# ---------------------------------------------------------------------------


class CognitionLog(Base):
    """Audit log of every cognition (prediction / optimisation) cycle.

    Each row records one pass through the discovery loop -- model
    inference, reward estimation, and any molecular modifications that
    were applied.
    """

    __tablename__ = "cognition_log"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to Discovery
    discovery_id: Mapped[int] = mapped_column(
        ForeignKey("discovery.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Cycle metadata
    cycle_number: Mapped[int] = mapped_column(Integer, nullable=False)
    cycle_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # e.g. "prediction", "optimisation", "validation"

    # Model information
    model_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    model_type: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )  # e.g. "chembl_gnn", "random_forest"

    # Input / output
    input_smiles: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    output_smiles: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Scores
    predicted_affinity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    predicted_affinity_unit: Mapped[str] = mapped_column(
        String(8), nullable=False, default="pIC50"
    )
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reward_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Modification details
    modification_type: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )  # e.g. "atom_substitution", "ring_expansion"
    modification_details: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )

    # Validation references
    validation_passed: Mapped[Optional[bool]] = mapped_column(
        Boolean, nullable=True
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    discovery: Mapped["Discovery"] = relationship(
        "Discovery", back_populates="cognition_logs"
    )

    def __repr__(self) -> str:
        return (
            f"<CognitionLog(id={self.id}, discovery_id={self.discovery_id}, "
            f"cycle={self.cycle_number}, type={self.cycle_type!r})>"
        )

    def to_dict(self) -> dict:
        """Serialise the cognition log entry to a plain dictionary."""
        return {
            "id": self.id,
            "discovery_id": self.discovery_id,
            "cycle_number": self.cycle_number,
            "cycle_type": self.cycle_type,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "input_smiles": self.input_smiles,
            "output_smiles": self.output_smiles,
            "predicted_affinity": self.predicted_affinity,
            "predicted_affinity_unit": self.predicted_affinity_unit,
            "confidence_score": self.confidence_score,
            "reward_score": self.reward_score,
            "modification_type": self.modification_type,
            "modification_details": self.modification_details,
            "validation_passed": self.validation_passed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
