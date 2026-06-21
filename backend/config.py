"""Application configuration using pydantic-settings.

All defaults are tuned for EGFR-targeted molecular discovery,
matching the specifications for the core engine pipeline.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the molecular discovery engine.

    Attributes:
        project_name: Human-readable project identifier.
        target_chembl_id: ChEMBL target ID (default: CHEMBL203 = EGFR).
        target_name: Common name of the biological target.
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
        data_dir: Directory for local data caches.
    """

    # Project identity
    project_name: str = "Molecular Discovery Engine"

    # Target configuration — EGFR is the default
    target_chembl_id: str = "CHEMBL203"
    target_name: str = "Epidermal Growth Factor Receptor (EGFR)"
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
