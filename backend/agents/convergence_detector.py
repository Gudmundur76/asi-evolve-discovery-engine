"""Convergence Detector — Cross-track Tanimoto analysis for the HIV-1 Protease engine.

Molecules appearing in 2+ tracks with Tanimoto similarity ≥ 0.7 are flagged as
convergence candidates. Convergence is the primary scientific signal: when
independent tracks with different seeding strategies converge on structurally
similar molecules, it is strong evidence that the scaffold is a genuine
HIV protease inhibitor pharmacophore.

Milestone schedule:
    Day 7:  First convergence analysis — threshold 0.70, min_tracks 2
    Day 15: Mid-point analysis — threshold 0.75, min_tracks 2
    Day 22: Pre-publication — threshold 0.80, min_tracks 2
    Day 30: Final publication — threshold 0.80, min_tracks 2, top 4–8 candidates

All results are persisted to the SQLite database and written to
data/convergence/ as JSON + Markdown reports.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Tanimoto thresholds per milestone day
MILESTONE_THRESHOLDS: Dict[int, float] = {
    7: 0.70,
    15: 0.75,
    22: 0.80,
    30: 0.80,
}

# Tracks in the 4-track architecture
TRACK_NAMES = {
    "A": "ChEMBL Top Actives",
    "B": "PDB Co-Crystal Ligands",
    "C": "BindingDB Curated",
    "D": "Diverse Scaffolds",
}


@dataclass
class ConvergenceCandidate:
    """A molecule that appears in 2+ tracks."""
    smiles: str
    tracks: List[str]                   # e.g. ["A", "C"]
    tanimoto_scores: List[float]        # pairwise scores
    mean_tanimoto: float
    best_pic50: float
    mean_pic50: float
    citation_confidence: float
    pubmed_ids: List[str] = field(default_factory=list)
    scaffold_family: str = ""
    day_detected: int = 0
    milestone: str = ""
    rank: int = 0


@dataclass
class ConvergenceReport:
    """Full convergence analysis report for a milestone day."""
    day: int
    milestone: str
    threshold: float
    candidates: List[ConvergenceCandidate]
    total_corpus_size: int
    tracks_analysed: int
    analysis_timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


class ConvergenceDetector:
    """Detects cross-track convergence candidates using Tanimoto similarity.

    Parameters
    ----------
    data_dir:
        Directory where convergence reports are written.
    min_tracks:
        Minimum number of tracks a molecule must appear in (default 2).
    """

    def __init__(
        self,
        data_dir: str = "data/convergence",
        min_tracks: int = 2,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.min_tracks = min_tracks
        logger.info(
            "ConvergenceDetector initialized: data_dir=%s, min_tracks=%d",
            data_dir, min_tracks,
        )

    def should_run(self, day: int) -> bool:
        """Return True if convergence analysis should run on this day."""
        return day in MILESTONE_THRESHOLDS or day >= 7

    def get_threshold(self, day: int) -> float:
        """Return the Tanimoto threshold for a given day."""
        if day in MILESTONE_THRESHOLDS:
            return MILESTONE_THRESHOLDS[day]
        # For days between milestones, use the previous milestone threshold
        for milestone_day in sorted(MILESTONE_THRESHOLDS.keys(), reverse=True):
            if day >= milestone_day:
                return MILESTONE_THRESHOLDS[milestone_day]
        return 0.70

    def get_milestone_label(self, day: int) -> str:
        """Return a human-readable milestone label."""
        if day <= 7:
            return "Day 7 — First Convergence Analysis"
        elif day <= 15:
            return "Day 15 — Mid-Point Analysis"
        elif day <= 22:
            return "Day 22 — Pre-Publication"
        else:
            return "Day 30 — Final Publication"

    def analyse(
        self,
        corpus_records: List[Dict[str, Any]],
        day: int,
    ) -> ConvergenceReport:
        """Run convergence analysis on the current corpus.

        Parameters
        ----------
        corpus_records:
            List of dicts with at minimum: smiles, track, predicted_pic50,
            citation_confidence, pubmed_ids.
        day:
            Current day number (1–30).

        Returns
        -------
        ConvergenceReport with all convergence candidates found.
        """
        threshold = self.get_threshold(day)
        milestone = self.get_milestone_label(day)

        logger.info(
            "ConvergenceDetector: day=%d, threshold=%.2f, corpus=%d records",
            day, threshold, len(corpus_records),
        )

        # Group records by track
        tracks: Dict[str, List[Dict[str, Any]]] = {}
        for rec in corpus_records:
            track = rec.get("track", "A")
            if track not in tracks:
                tracks[track] = []
            tracks[track].append(rec)

        logger.info(
            "Tracks with records: %s",
            {t: len(recs) for t, recs in tracks.items()},
        )

        # Compute fingerprints for all records
        fps = self._compute_fingerprints(corpus_records)

        # Find cross-track pairs with Tanimoto ≥ threshold
        candidates = self._find_convergent(
            corpus_records, fps, tracks, threshold, day, milestone
        )

        # Sort by mean pIC50 descending
        candidates.sort(key=lambda c: c.mean_pic50, reverse=True)
        for i, cand in enumerate(candidates):
            cand.rank = i + 1

        report = ConvergenceReport(
            day=day,
            milestone=milestone,
            threshold=threshold,
            candidates=candidates,
            total_corpus_size=len(corpus_records),
            tracks_analysed=len(tracks),
        )

        # Persist report
        self._save_report(report)

        logger.info(
            "ConvergenceDetector: found %d convergence candidates on day %d",
            len(candidates), day,
        )
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_fingerprints(
        self, records: List[Dict[str, Any]]
    ) -> Dict[int, Any]:
        """Compute Morgan fingerprints for all records. Returns {idx: fp}."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, DataStructs
        except ImportError:
            logger.error("RDKit not available — cannot compute Tanimoto similarity")
            return {}

        fps: Dict[int, Any] = {}
        for i, rec in enumerate(records):
            smiles = rec.get("smiles", "")
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fps[i] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return fps

    def _tanimoto(self, fp1: Any, fp2: Any) -> float:
        """Compute Tanimoto similarity between two RDKit fingerprints."""
        try:
            from rdkit.Chem import DataStructs
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except Exception:
            return 0.0

    def _find_convergent(
        self,
        records: List[Dict[str, Any]],
        fps: Dict[int, Any],
        tracks: Dict[str, List[Dict[str, Any]]],
        threshold: float,
        day: int,
        milestone: str,
    ) -> List[ConvergenceCandidate]:
        """Find molecules that appear in 2+ tracks above the Tanimoto threshold."""
        if len(fps) < 2:
            return []

        # Build index: record_idx → track
        idx_to_track: Dict[int, str] = {}
        for i, rec in enumerate(records):
            idx_to_track[i] = rec.get("track", "A")

        # Find all pairs across different tracks with Tanimoto ≥ threshold
        # Group by representative molecule (the one with highest pIC50 in the cluster)
        clusters: List[Dict[str, Any]] = []
        used: set = set()

        indices = sorted(fps.keys())
        for i in range(len(indices)):
            idx_i = indices[i]
            if idx_i in used:
                continue
            track_i = idx_to_track[idx_i]
            cluster_tracks = {track_i}
            cluster_indices = [idx_i]
            tanimoto_scores = []

            for j in range(i + 1, len(indices)):
                idx_j = indices[j]
                track_j = idx_to_track[idx_j]
                if track_j == track_i:
                    continue  # Same track — skip
                sim = self._tanimoto(fps[idx_i], fps[idx_j])
                if sim >= threshold:
                    cluster_tracks.add(track_j)
                    cluster_indices.append(idx_j)
                    tanimoto_scores.append(sim)

            if len(cluster_tracks) >= self.min_tracks:
                # This is a convergence candidate
                cluster_records = [records[k] for k in cluster_indices]
                best_rec = max(
                    cluster_records,
                    key=lambda r: r.get("predicted_pic50", 0.0),
                )
                mean_pic50 = sum(
                    r.get("predicted_pic50", 0.0) for r in cluster_records
                ) / len(cluster_records)
                mean_tanimoto = (
                    sum(tanimoto_scores) / len(tanimoto_scores)
                    if tanimoto_scores else threshold
                )
                # Collect all PubMed IDs from cluster
                pubmed_ids: List[str] = []
                for r in cluster_records:
                    pubmed_ids.extend(r.get("pubmed_ids", []))
                pubmed_ids = list(dict.fromkeys(pubmed_ids))  # deduplicate, preserve order

                candidate = ConvergenceCandidate(
                    smiles=best_rec.get("smiles", ""),
                    tracks=sorted(cluster_tracks),
                    tanimoto_scores=tanimoto_scores,
                    mean_tanimoto=mean_tanimoto,
                    best_pic50=best_rec.get("predicted_pic50", 0.0),
                    mean_pic50=mean_pic50,
                    citation_confidence=best_rec.get("citation_confidence", 0.0),
                    pubmed_ids=pubmed_ids,
                    scaffold_family=best_rec.get("scaffold_family", ""),
                    day_detected=day,
                    milestone=milestone,
                )
                clusters.append(candidate)
                for k in cluster_indices:
                    used.add(k)

        return clusters

    def _save_report(self, report: ConvergenceReport) -> None:
        """Save the convergence report as JSON and Markdown."""
        day_str = f"day{report.day:02d}"
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # JSON
        json_path = self.data_dir / f"convergence_{day_str}_{ts}.json"
        report_dict = {
            "day": report.day,
            "milestone": report.milestone,
            "threshold": report.threshold,
            "total_corpus_size": report.total_corpus_size,
            "tracks_analysed": report.tracks_analysed,
            "analysis_timestamp": report.analysis_timestamp,
            "candidates": [
                {
                    "rank": c.rank,
                    "smiles": c.smiles,
                    "tracks": c.tracks,
                    "mean_tanimoto": round(c.mean_tanimoto, 4),
                    "best_pic50": round(c.best_pic50, 4),
                    "mean_pic50": round(c.mean_pic50, 4),
                    "citation_confidence": round(c.citation_confidence, 4),
                    "pubmed_ids": c.pubmed_ids,
                    "scaffold_family": c.scaffold_family,
                    "day_detected": c.day_detected,
                    "milestone": c.milestone,
                }
                for c in report.candidates
            ],
        }
        with open(json_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        # Markdown summary
        md_path = self.data_dir / f"convergence_{day_str}_{ts}.md"
        lines = [
            f"# Convergence Report — {report.milestone}",
            f"",
            f"**Day:** {report.day}  ",
            f"**Tanimoto threshold:** {report.threshold}  ",
            f"**Corpus size:** {report.total_corpus_size}  ",
            f"**Tracks analysed:** {report.tracks_analysed}  ",
            f"**Convergence candidates:** {len(report.candidates)}  ",
            f"**Timestamp:** {report.analysis_timestamp}  ",
            f"",
            f"## Candidates",
            f"",
        ]
        if not report.candidates:
            lines.append("No convergence candidates found at this threshold.")
        else:
            for c in report.candidates:
                track_str = " + ".join(
                    f"Track {t} ({TRACK_NAMES.get(t, t)})" for t in c.tracks
                )
                lines += [
                    f"### Rank {c.rank}",
                    f"",
                    f"- **SMILES:** `{c.smiles}`",
                    f"- **Tracks:** {track_str}",
                    f"- **Best pIC50:** {c.best_pic50:.4f}",
                    f"- **Mean pIC50:** {c.mean_pic50:.4f}",
                    f"- **Mean Tanimoto:** {c.mean_tanimoto:.4f}",
                    f"- **Citation confidence:** {c.citation_confidence:.4f}",
                    f"- **PubMed IDs:** {', '.join(c.pubmed_ids) if c.pubmed_ids else 'None'}",
                    f"- **Scaffold family:** {c.scaffold_family or 'Unknown'}",
                    f"",
                ]

        with open(md_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(
            "ConvergenceDetector: report saved to %s and %s",
            json_path, md_path,
        )
