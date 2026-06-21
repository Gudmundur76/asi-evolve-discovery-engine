"""ChEMBL REST API client with pagination, retry logic, and error handling.

Provides a high-level interface for fetching bioactivity data and target
metadata from the ChEMBL database.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from backend.config import settings

logger = logging.getLogger(__name__)


class ChEMBLClientError(Exception):
    """Base exception for ChEMBL client errors."""

    pass


class ChEMBLAPIError(ChEMBLClientError):
    """Raised when the ChEMBL API returns an error response."""

    pass


class ChEMBLClient:
    """Client for the ChEMBL REST API.

    Fetches bioactivity records and target metadata with automatic
    pagination, deduplication, and exponential-backoff retries.

    Args:
        target_chembl_id: ChEMBL target identifier (default from settings).
        base_url: ChEMBL API base URL (default from settings).
        max_retries: Maximum retry attempts per request (default from settings).
        backoff_factor: Exponential backoff multiplier (default from settings).
    """

    def __init__(
        self,
        target_chembl_id: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: Optional[int] = None,
        backoff_factor: Optional[float] = None,
    ) -> None:
        self.target_chembl_id = target_chembl_id or settings.target_chembl_id
        self.base_url = base_url or settings.chembl_base_url
        self.max_retries = max_retries or settings.max_retries
        self.backoff_factor = backoff_factor or settings.backoff_factor
        self._session = requests.Session()
        # ChEMBL API requests a user-agent; set a descriptive one
        self._session.headers.update(
            {"User-Agent": "molecular-discovery-engine/0.1.0 (dev@localhost)"}
        )

    def _paginated_get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Execute a GET with ChEMBL-style limit/offset pagination.

        Args:
            endpoint: API endpoint path (e.g. 'activity.json').
            params: Query parameters merged into every request.
            limit: Page size (ChEMBL max is typically 1000).

        Returns:
            Combined list of all record dictionaries across all pages.

        Raises:
            ChEMBLAPIError: If the API returns a non-OK status after all retries.
        """
        url = f"{self.base_url}/{endpoint}"
        params = dict(params or {})
        params["limit"] = limit
        params["offset"] = 0

        all_records: List[Dict[str, Any]] = []

        while True:
            attempt = 0
            response = None

            while attempt <= self.max_retries:
                try:
                    response = self._session.get(url, params=params, timeout=60)
                    if response.status_code == 200:
                        break
                    # 429 = rate limited; 5xx = transient server error
                    if response.status_code in (429, 500, 502, 503, 504):
                        wait = self.backoff_factor * (2 ** attempt)
                        logger.warning(
                            "ChEMBL %s (attempt %d/%d), retrying in %.1fs ...",
                            response.status_code,
                            attempt + 1,
                            self.max_retries,
                            wait,
                        )
                        time.sleep(wait)
                        attempt += 1
                        continue
                    # Non-retryable status code
                    raise ChEMBLAPIError(
                        f"ChEMBL API error {response.status_code}: {response.text[:500]}"
                    )
                except requests.exceptions.RequestException as exc:
                    wait = self.backoff_factor * (2 ** attempt)
                    logger.warning(
                        "ChEMBL request exception (attempt %d/%d): %s. "
                        "Retrying in %.1fs ...",
                        attempt + 1,
                        self.max_retries,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                    attempt += 1

            if response is None or response.status_code != 200:
                raise ChEMBLAPIError(
                    f"Failed to fetch {url} after {self.max_retries + 1} attempts"
                )

            data = response.json()
            page = data.get("activities", []) if "activities" in data else [data] if not isinstance(data, list) else data
            # Some endpoints return a single object under 'target' etc.
            if isinstance(data, dict) and endpoint.endswith(".json"):
                # Check common wrapper keys
                for key in ("activities", "targets", "molecules"):
                    if key in data:
                        page = data[key]
                        break

            if not page:
                break

            all_records.extend(page)

            # ChEMBL signals the end via pageMeta or when fewer records than limit
            page_meta = data.get("pageMeta", {}) if isinstance(data, dict) else {}
            total_count = page_meta.get("totalCount")
            if total_count is not None and len(all_records) >= total_count:
                break
            if len(page) < limit:
                break

            params["offset"] += limit

        return all_records

    def fetch_activities(
        self,
        activity_type: str = "IC50",
        limit: int = 5000,
    ) -> pd.DataFrame:
        """Fetch bioactivity records for the configured target.

        Retrieves activities filtered by *target_chembl_id*,
        *standard_type*, and *standard_units=nM*.  Removes rows with
        missing SMILES or missing *standard_value*, deduplicates by
        *molecule_chembl_id* (keeping the lowest / best affinity), and
        converts *standard_value* to float (affinity in nM).

        Args:
            activity_type: Measurement type, e.g. "IC50", "Ki", "EC50".
            limit: Maximum total records to fetch.

        Returns:
            DataFrame with columns
            ``[molecule_chembl_id, canonical_smiles, standard_value, standard_units]``.
        """
        logger.info(
            "Fetching %s activities for target %s ...",
            activity_type,
            self.target_chembl_id,
        )

        records = self._paginated_get(
            endpoint="activity.json",
            params={
                "target_chembl_id": self.target_chembl_id,
                "standard_type": activity_type,
                "standard_units": "nM",
            },
            limit=min(limit, 1000),
        )

        if not records:
            logger.warning("No activities returned from ChEMBL")
            return pd.DataFrame(
                columns=[
                    "molecule_chembl_id",
                    "canonical_smiles",
                    "standard_value",
                    "standard_units",
                ]
            )

        df = pd.DataFrame(records)

        # Keep only rows with both SMILES and a numeric value
        required_cols = ["molecule_chembl_id", "canonical_smiles", "standard_value"]
        for col in required_cols:
            if col not in df.columns:
                raise ChEMBLClientError(
                    f"Expected column '{col}' missing from ChEMBL response"
                )

        df = df.dropna(subset=["canonical_smiles", "standard_value"]).copy()
        df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
        df = df.dropna(subset=["standard_value"])

        # Keep only the required columns
        df = df[
            [
                "molecule_chembl_id",
                "canonical_smiles",
                "standard_value",
                "standard_units",
            ]
        ].copy()

        # Deduplicate: keep the lowest standard_value (best affinity) per molecule
        df = df.sort_values("standard_value").drop_duplicates(
            subset=["molecule_chembl_id"], keep="first"
        )

        logger.info(
            "Fetched %d unique activities for target %s",
            len(df),
            self.target_chembl_id,
        )
        return df.reset_index(drop=True)

    def fetch_target_info(self) -> Dict[str, str]:
        """Fetch metadata for the configured target.

        Returns:
            Dictionary with keys ``target_name``, ``organism``, ``target_type``.
        """
        logger.info("Fetching target info for %s ...", self.target_chembl_id)

        records = self._paginated_get(
            endpoint="target.json",
            params={"target_chembl_id": self.target_chembl_id},
            limit=10,
        )

        if not records:
            raise ChEMBLClientError(
                f"No target info found for {self.target_chembl_id}"
            )

        info = records[0]
        return {
            "target_name": info.get("pref_name", "Unknown"),
            "organism": info.get("organism", "Unknown"),
            "target_type": info.get("target_type", "Unknown"),
        }
