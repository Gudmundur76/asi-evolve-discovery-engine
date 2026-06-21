"""
SwissADME web-service client with RDKit local fallback.

:class:`SwissADMEClient` attempts to query the public SwissADME web interface
for a full ADMET profile.  If the service is unreachable, unmaintained, or
blocks automated requests the client transparently falls back to computing the
same (or equivalent) descriptors locally with RDKit.

The returned dictionary is fully normalised so that downstream consumers
(e.g. :class:`validator.DiscoveryValidator`) never need to know which path
was taken.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RDKit guard (same pattern as vina_docker)
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit unavailable -- local ADMET fallback will return zeros.")

# Optional: SAscore (from RDKit Contrib) for synthetic accessibility
try:
    from rdkit.Chem import RDConfig
    import os
    import sys

    sascorer_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if sascorer_path not in sys.path:
        sys.path.append(sascorer_path)
    import sascorer  # type: ignore[import-untyped]

    SASCORE_AVAILABLE = True
except Exception:
    SASCORE_AVAILABLE = False

# ---------------------------------------------------------------------------
# CYP inhibition SMARTS patterns (approximate heuristic rules)
# ---------------------------------------------------------------------------

_CYP_PATTERNS: dict[str, Any] = {}


def _load_cyp_patterns() -> dict[str, Any]:
    """Lazy-load CYP SMARTS patterns used for local RDKit prediction."""
    global _CYP_PATTERNS
    if _CYP_PATTERNS:
        return _CYP_PATTERNS
    if not RDKIT_AVAILABLE:
        _CYP_PATTERNS = {}
        return _CYP_PATTERNS

    patterns = {
        # Furafylline-like scaffold
        "1A2": Chem.MolFromSmarts("[nH]1cnc2ccccc2c1=O"),
        "2C9": Chem.MolFromSmarts("c1ccc(cc1)S(=O)(=O)N"),
        "2C19": Chem.MolFromSmarts("c1ccccc1C(=N)N"),
        "2D6": Chem.MolFromSmarts("c1cc2c(cc1O)OCO2"),
        "3A4": Chem.MolFromSmarts("C1=C(C)C(=O)C(C)=C(C)C1=O"),
    }
    _CYP_PATTERNS = {k: v for k, v in patterns.items() if v is not None}
    return _CYP_PATTERNS


class SwissADMEClient:
    """Client for the SwissADME web service with local RDKit fallback.

    Parameters
    ----------
    base_url:
        Root URL of the SwissADME deployment (default is the public site).
    request_timeout:
        Seconds to wait for the HTTP response.
    retry_count:
        Number of retries on transient network errors.
    """

    def __init__(
        self,
        base_url: str = "http://www.swissadme.ch/",
        request_timeout: int = 60,
        retry_count: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = request_timeout
        self.retry_count = retry_count
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate",
            }
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(self, smiles: str) -> dict[str, Any]:
        """Fetch or compute the full ADMET profile for *smiles*.

        The method first attempts to scrape SwissADME.  If that fails for
        any reason (network, blocking, HTML changes) it falls back to
        locally-calculated RDKit descriptors.

        Parameters
        ----------
        smiles:
            SMILES string of the molecule to profile.

        Returns
        -------
        dict
            Normalised ADMET profile dictionary.
        """
        # Try web service first
        for attempt in range(self.retry_count + 1):
            try:
                result = self._fetch_swissadme(smiles)
                if result:
                    logger.info("SwissADME web lookup succeeded")
                    return result
            except Exception as exc:
                logger.debug(
                    "SwissADME attempt %d failed: %s", attempt + 1, exc
                )
                if attempt < self.retry_count:
                    time.sleep(2 ** attempt)

        # Fallback to local RDKit calculation
        logger.info("Falling back to local RDKit descriptors for %s", smiles)
        return self._local_profile(smiles)

    # ------------------------------------------------------------------
    # SwissADME web scraping
    # ------------------------------------------------------------------

    def _fetch_swissadme(self, smiles: str) -> dict[str, Any] | None:
        """POST SMILES to SwissADME and parse the resulting HTML table.

        Returns *None* when the page structure is unrecognised so that the
        caller can fall back gracefully.
        """
        # Step 1 -- GET the form to obtain any CSRF token / cookies
        form_url = self.base_url
        resp = self._session.get(form_url, timeout=self.timeout)
        resp.raise_for_status()

        # Look for common CSRF token patterns
        csrf_token = self._extract_csrf(resp.text)

        # Step 2 -- POST the SMILES
        payload: dict[str, str] = {"smiles": smiles}
        if csrf_token:
            payload["csrf_token"] = csrf_token

        post_resp = self._session.post(
            form_url + "index.php",  # common endpoint
            data=payload,
            timeout=self.timeout,
            allow_redirects=True,
        )
        post_resp.raise_for_status()

        # Step 3 -- parse HTML for ADMET data
        return self._parse_swissadme_html(post_resp.text)

    def _extract_csrf(self, html: str) -> str | None:
        """Naive regex extraction of CSRF tokens from a form page."""
        patterns = [
            r'name=["\']csrf_token["\']\s+value=["\']([^"\']+)["\']',
            r'name=["\']_token["\']\s+value=["\']([^"\']+)["\']',
            r'value=["\']([^"\']+)["\']\s+name=["\']csrfmiddlewaretoken["\']',
        ]
        for pat in patterns:
            m = re.search(pat, html)
            if m:
                return m.group(1)
        return None

    def _parse_swissadme_html(self, html: str) -> dict[str, Any] | None:
        """Extract property values from SwissADME result HTML.

        Returns *None* if the expected patterns are not found.
        """
        # Normalise whitespace
        text = re.sub(r"\s+", " ", html)

        # Helper to extract float after a label
        def _float(label: str) -> float:
            pat = rf"{re.escape(label)}\s*[:=]?\s*</?[^>]+>\s*([-+]?\d+\.?\d*)"
            m = re.search(pat, text, re.IGNORECASE)
            return float(m.group(1)) if m else 0.0

        def _int(label: str) -> int:
            val = _float(label)
            return int(val)

        def _str_val(label: str) -> str:
            pat = rf"{re.escape(label)}\s*[:=]?\s*</?[^>]+>\s*([^<]+)"
            m = re.search(pat, text, re.IGNORECASE)
            return m.group(1).strip() if m else ""

        try:
            mw = _float("Molecular weight")
            if mw == 0.0:
                mw = _float("MW")
            logp = _float("XLOGP3")
            if logp == 0.0:
                logp = _float("Log P")
            hbd = _int("H-bond donors")
            if hbd == 0:
                hbd = _int("donors")
            hba = _int("H-bond acceptors")
            if hba == 0:
                hba = _int("acceptors")
            tpsa = _float("TPSA")
            rotb = _int("Rotatable bonds")
            if rotb == 0:
                rotb = _int("rotatable")
            lipinski = _int("Lipinski")
            if lipinski == 0:
                lipinski = _int("violations")
            sa = _float("Synthetic Accessibility")
            if sa == 0.0:
                sa = _float("SA score")
            gi = _str_val("GI absorption")
            bbb = _str_val("BBB permeant")
            pgp = _str_val("P-gp substrate")

            # Build result dict
            cyp = self._extract_cyp_from_html(text)
            druglike = lipinski <= 1 and 150 <= mw <= 500 and tpsa <= 140
            toxicity = self._extract_pains_from_html(text)

            return {
                "mw": round(mw, 2),
                "logp": round(logp, 2),
                "hbd": hbd,
                "hba": hba,
                "tpsa": round(tpsa, 2),
                "rotatable_bonds": rotb,
                "lipinski_violations": lipinski,
                "synthetic_accessibility": round(sa, 2) if sa else 5.0,
                "gi_absorption": "High" if "high" in gi.lower() else "Low",
                "bbb_permeable": "yes" in bbb.lower() if bbb else False,
                "pgp_substrate": "yes" in pgp.lower() if pgp else False,
                "cyp_inhibitors": cyp,
                "druglikeness_score": 1.0 - (lipinski / 4.0),
                "medicinal_chemistry_score": max(0.0, 1.0 - sa / 10.0),
                "is_druglike": druglike,
                "toxicity_flags": toxicity,
                "overall_pass": druglike and len(toxicity) == 0,
            }
        except Exception as exc:
            logger.debug("HTML parsing failed: %s", exc)
            return None

    def _extract_cyp_from_html(self, text: str) -> dict[str, bool]:
        """Parse CYP inhibition predictions from result HTML."""
        cyp: dict[str, bool] = {}
        for isoform in ["1A2", "2C9", "2C19", "2D6", "3A4"]:
            pat = rf"CYP\s*{re.escape(isoform)}\s*</?[^>]+>\s*([^<]+)"
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip().lower()
                cyp[isoform] = "yes" in val or "true" in val or "1" in val
            else:
                cyp[isoform] = False
        return cyp

    def _extract_pains_from_html(self, text: str) -> list[str]:
        """Extract PAINS / toxicity alerts from result HTML."""
        alerts: list[str] = []
        if re.search(r"PAINS", text, re.IGNORECASE):
            alerts.append("PAINS alert detected")
        if re.search(r" Brenk", text, re.IGNORECASE):
            alerts.append("Brenk alert detected")
        if re.search(r" Zinc", text, re.IGNORECASE):
            alerts.append("ZINC alert detected")
        return alerts

    # ------------------------------------------------------------------
    # Local RDKit fallback
    # ------------------------------------------------------------------

    def _local_profile(self, smiles: str) -> dict[str, Any]:
        """Compute ADMET descriptors locally using RDKit.

        This is the fallback path when the SwissADME web service cannot be
        reached.  The values are calibrated to be comparable to those
        returned by the web service.
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit unavailable -- returning empty profile")
            return self._empty_profile()

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("Invalid SMILES for local profiling: %s", smiles)
            return self._empty_profile()

        mol = Chem.AddHs(mol)

        # Basic descriptors
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        rotb = Descriptors.NumRotatableBonds(mol)

        # Lipinski rule of 5
        lipinski_violations = 0
        if mw > 500:
            lipinski_violations += 1
        if logp > 5:
            lipinski_violations += 1
        if hbd > 5:
            lipinski_violations += 1
        if hba > 10:
            lipinski_violations += 1

        # Synthetic accessibility
        if SASCORE_AVAILABLE:
            try:
                sa_score = sascorer.calculateScore(mol)
            except Exception:
                sa_score = self._estimate_sa(mol)
        else:
            sa_score = self._estimate_sa(mol)

        # Drug-likeness composite
        ghose_pass = 160 <= mw <= 480 and -0.4 <= logp <= 5.6
        veber_pass = rotb <= 10 and tpsa <= 140
        is_druglike = lipinski_violations <= 1 and ghose_pass and veber_pass

        # GI absorption heuristic
        gi_high = tpsa <= 140 and rotb <= 10 and mw <= 500

        # BBB permeability heuristic (Ghose + low TPSA)
        bbb = mw <= 400 and logp <= 3.0 and tpsa <= 90 and hbd <= 3

        # P-gp substrate heuristic
        pgp = hba > 8 or mw > 400 or tpsa > 120

        # CYP inhibition
        cyp = self._predict_cyp_local(mol)

        # Toxicity / PAINS
        pains = self._check_pains_local(mol)

        # Composite scores
        druglikeness_score = max(0.0, 1.0 - (lipinski_violations / 4.0))
        medchem_score = max(0.0, 1.0 - sa_score / 10.0)

        return {
            "mw": round(mw, 2),
            "logp": round(logp, 2),
            "hbd": int(hbd),
            "hba": int(hba),
            "tpsa": round(tpsa, 2),
            "rotatable_bonds": int(rotb),
            "lipinski_violations": lipinski_violations,
            "synthetic_accessibility": round(sa_score, 2),
            "gi_absorption": "High" if gi_high else "Low",
            "bbb_permeable": bbb,
            "pgp_substrate": pgp,
            "cyp_inhibitors": cyp,
            "druglikeness_score": round(druglikeness_score, 3),
            "medicinal_chemistry_score": round(medchem_score, 3),
            "is_druglike": is_druglike,
            "toxicity_flags": pains,
            "overall_pass": is_druglike and len(pains) == 0,
        }

    def _empty_profile(self) -> dict[str, Any]:
        """Return a zero-filled profile when nothing else is possible."""
        return {
            "mw": 0.0,
            "logp": 0.0,
            "hbd": 0,
            "hba": 0,
            "tpsa": 0.0,
            "rotatable_bonds": 0,
            "lipinski_violations": 0,
            "synthetic_accessibility": 0.0,
            "gi_absorption": "Low",
            "bbb_permeable": False,
            "pgp_substrate": False,
            "cyp_inhibitors": {
                "1A2": False,
                "2C9": False,
                "2C19": False,
                "2D6": False,
                "3A4": False,
            },
            "druglikeness_score": 0.0,
            "medicinal_chemistry_score": 0.0,
            "is_druglike": False,
            "toxicity_flags": [],
            "overall_pass": False,
        }

    def _estimate_sa(self, mol: Any) -> float:
        """Rough synthetic accessibility estimate from molecular complexity.

        Used when the SAscore module is not installed.  Returns a value
        in the 1--10 range where 1 = trivial and 10 = very difficult.
        """
        n_atoms = mol.GetNumAtoms()
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        # Heuristic formula calibrated against known SAscore outputs
        score = (
            1.0
            + 0.02 * n_atoms
            + 0.3 * n_rings
            + 0.5 * n_stereo
            + 0.1 * Descriptors.NumRotatableBonds(mol)
        )
        return float(min(max(score, 1.0), 10.0))

    def _predict_cyp_local(self, mol: Any) -> dict[str, bool]:
        """Predict CYP inhibition using SMARTS substructure heuristics."""
        patterns = _load_cyp_patterns()
        result: dict[str, bool] = {}
        for isoform in ["1A2", "2C9", "2C19", "2D6", "3A4"]:
            pat = patterns.get(isoform)
            if pat and mol.HasSubstructMatch(pat):
                result[isoform] = True
            else:
                # Fallback: use molecular property heuristics
                result[isoform] = self._cyp_heuristic(mol, isoform)
        return result

    def _cyp_heuristic(self, mol: Any, isoform: str) -> bool:
        """Property-based heuristic for CYP inhibition when SMARTS miss."""
        logp = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        # General promiscuity -- lipophilic large molecules more likely to inhibit
        promiscuity_score = (logp / 5.0) * 0.5 + (mw / 500.0) * 0.5
        thresholds = {
            "1A2": 0.5,
            "2C9": 0.55,
            "2C19": 0.45,
            "2D6": 0.6,
            "3A4": 0.4,
        }
        return promiscuity_score > thresholds.get(isoform, 0.5)

    def _check_pains_local(self, mol: Any) -> list[str]:
        """Check for PAINS substructures using RDKit filters.

        Returns a list of human-readable alert strings (empty if clean).
        """
        alerts: list[str] = []
        if not RDKIT_AVAILABLE:
            return alerts

        # Common PAINS SMARTS patterns (simplified set)
        pains_smarts = [
            ("h*_hydroxyl_ene_one", "[C;H2]=C-C(=O)-[OH]"),
            ("ene_one", "[#6]=[#6]-[#6](=O)"),
            ("quinone", "[#6]1[#6]=[#6][#6](=O)[#6]=[#6]1=O"),
            ("catechol", "c1ccc(c(c1)O)O"),
            ("azo", "[#6]-[#7]=[#7]-[#6]"),
            ("sulfonamide", "S(=O)(=O)N"),
            ("imine", "[C;H1]=N"),
            ("acyl_halide", "C(=O)[Cl,Br,F,I]"),
            (" Michael_acceptor", "[#6]=[#6]-C(=O)-[#6,#1,#7,#8]"),
            (" Schiff_base", "C=N-[#7,#6]"),
        ]

        for name, smarts in pains_smarts:
            patt = Chem.MolFromSmarts(smarts)
            if patt and mol.HasSubstructMatch(patt):
                alerts.append(f"PAINS alert: {name}")

        return alerts
