"""Application configuration using pydantic-settings.

Retargeted from EGFR (CHEMBL203) to HIV-1 Protease (CHEMBL2094253).
All new settings for citation verification, convergence detection,
loop scheduling, and persistent logging are added here.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the HIV-1 Protease discovery engine.

    Attributes:
        project_name: Human-readable project identifier.
        target_chembl_id: ChEMBL target ID (HIV-1 Protease = CHEMBL2094253).
        target_name: Common name of the biological target.
        target_uniprot: UniProt accession for the target.
        activity_type: Bioactivity measurement type (e.g. IC50, EC50, Ki).
        activity_limit: Maximum number of activity records to fetch.
        fingerprint_radius: Morgan fingerprint radius (2 = ECFP4 equivalent).
        fingerprint_nbits: Length of the fingerprint bit vector.
        model_type: Default ML model type.
        model_path: Filesystem path for the trained model pickle.
        model_metadata_path: Filesystem path for training metadata JSON.
        n_estimators: Number of trees in the Random Forest.
        max_depth: Maximum depth of each tree.
        test_size: Fraction of data held out for testing.
        random_state: Seed for reproducible train/test splits.
        chembl_base_url: Base URL for the ChEMBL REST API.
        max_retries: Maximum number of API retry attempts.
        backoff_factor: Exponential backoff multiplier for retries.
        citation_is_url: Base URL for the citation.manus.space verification API.
        citation_is_vertical: Vertical name for HIV protease claims.
        citation_confidence_threshold: Minimum confidence to pass Gate 1.
        convergence_tanimoto_threshold: Tanimoto similarity threshold for cross-track convergence.
        convergence_min_tracks: Minimum number of tracks a molecule must appear in to be a convergence candidate.
        cycle_interval_seconds: Seconds between optimization cycles (default 72 min).
        max_cycles: Maximum cycles to run (0 = unlimited).
        affinity_threshold_nm: Trigger experimental validation below this affinity (nM).
        persistent_drive_repo: GitHub repo slug for manus-persistent-drive logging.
        data_dir: Directory for local data caches.
    """

    # Project identity
    project_name: str = "HIV Protease Discovery Engine"

    # Target configuration — HIV-1 Protease
    target_chembl_id: str = "CHEMBL2094253"
    target_name: str = "HIV-1 Protease"
    target_uniprot: str = "P04585"
    activity_type: str = "IC50"
    activity_limit: int = 5000

    # Fingerprint configuration — 2048-bit Morgan (ECFP4 equivalent)
    fingerprint_radius: int = 2
    fingerprint_nbits: int = 2048

    # Model configuration
    model_type: str = "random_forest"
    model_path: Path = Field(default_factory=lambda: Path("data/model.pkl"))
    model_metadata_path: Path = Field(
        default_factory=lambda: Path("data/model_metadata.json")
    )
    n_estimators: int = 200
    max_depth: int = 20
    test_size: float = 0.2
    random_state: int = 42

    # API configuration
    chembl_base_url: str = "https://www.ebi.ac.uk/chembl/api/data"
    max_retries: int = 5
    backoff_factor: float = 1.5

    # Citation verification gate (Gate 1 — before ADMET and docking)
    citation_is_url: str = "https://citation.manus.space"
    citation_is_vertical: str = "hiv_protease"
    citation_confidence_threshold: float = 0.85

    # Convergence detection
    convergence_tanimoto_threshold: float = 0.70
    convergence_min_tracks: int = 2

    # Loop scheduling
    cycle_interval_seconds: int = 4320   # 72 min = 20 cycles/day
    max_cycles: int = 0                  # 0 = unlimited
    affinity_threshold_nm: float = 10.0  # trigger validation below this

    # Persistent logging
    persistent_drive_repo: str = "Gudmundur76/manus-persistent-drive"

    # Directories
    data_dir: Path = Field(default_factory=lambda: Path("data"))

    model_config = ConfigDict(
        env_prefix="MDE_",
        env_file=".env",
        env_file_encoding="utf-8",
        arbitrary_types_allowed=True,
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Resolve model paths relative to data_dir if not absolute
        if not self.model_path.is_absolute():
            self.model_path = self.data_dir / self.model_path.name
        if not self.model_metadata_path.is_absolute():
            self.model_metadata_path = self.data_dir / self.model_metadata_path.name


# Global settings singleton
settings = Settings()
