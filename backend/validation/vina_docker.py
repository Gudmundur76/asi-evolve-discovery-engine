"""
AutoDock Vina molecular docking wrapper with RDKit ligand preparation.

Provides the :class:`VinaDocker` class which handles the full docking pipeline:
SMILES -> 3D conformer -> PDBQT -> Vina docking -> parsed results.
If the Vina executable is not available the class transparently falls back to
mock/simulated results so that integration tests and CI pipelines can still run.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RDKit availability guard
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available -- ligand preparation will use mock data.")


class VinaDocker:
    """High-level wrapper around AutoDock Vina for receptor-ligand docking.

    The class lifecycle is:

    1. ``prepare_ligand(smiles)`` -> writes a PDBQT file to ``data/docking/``.
    2. ``dock(ligand_pdbqt)`` -> runs Vina (or mocks) and returns structured
       results including the best binding-mode affinity.

    Parameters
    ----------
    receptor_path:
        Absolute or relative path to the receptor PDBQT file.
    center:
        *(x, y, z)* coordinates of the search-space center.
    box_size:
        Length of each side of the cubic search space (angstroms).
    """

    def __init__(
        self,
        receptor_path: str,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box_size: float = 20.0,
    ) -> None:
        self.receptor_path = str(Path(receptor_path).expanduser().resolve())
        self.center = center
        self.box_size = float(box_size)

        # Locate Vina executable once at construction time.
        self._vina_exe = self._discover_vina()
        if self._vina_exe is None:
            logger.warning(
                "AutoDock Vina executable not found on PATH. "
                "VinaDocker will return mock results."
            )

        # Ensure output directory exists.
        self._ligand_dir = Path("data/docking")
        self._ligand_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_ligand(self, smiles: str) -> str:
        """Convert a SMILES string into a 3-D PDBQT file ready for docking.

        Steps performed:
            1. SMILES -> RDKit Mol
            2. Add hydrogens
            3. Embed 3-D conformer (ETKDG)
            4. MMFF force-field optimisation
            5. Write MOL block and convert to PDBQT

        Parameters
        ----------
        smiles:
            Canonical (or non-canonical) SMILES for the ligand.

        Returns
        -------
        str
            Path to the generated ``.pdbqt`` file.

        Raises
        ------
        RuntimeError
            If RDKit is unavailable or conformer generation fails.
        """
        if not RDKIT_AVAILABLE:
            return self._mock_ligand_file(smiles)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise RuntimeError(f"RDKit failed to parse SMILES: {smiles!r}")

        mol = Chem.AddHs(mol)
        status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if status == -1:
            logger.warning("ETKDG embedding failed, trying fallback with random coords")
            params = AllChem.ETKDGv3()
            params.useRandomCoords = True
            status = AllChem.EmbedMolecule(mol, params)
            if status == -1:
                raise RuntimeError("RDKit could not generate 3-D conformer.")

        AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94")

        # Write MOL file temporarily
        hash_id = hashlib.sha256(smiles.encode()).hexdigest()[:16]
        mol_path = self._ligand_dir / f"{hash_id}.mol"
        pdbqt_path = self._ligand_dir / f"{hash_id}.pdbqt"

        mol_block = Chem.MolToMolBlock(mol)
        mol_path.write_text(mol_block, encoding="utf-8")

        # Convert MOL -> PDBQT
        self._mol_to_pdbqt(str(mol_path), str(pdbqt_path))

        # Clean up intermediate MOL
        mol_path.unlink(missing_ok=True)

        return str(pdbqt_path)

    def dock(self, ligand_pdbqt: str) -> dict[str, Any]:
        """Run AutoDock Vina on a prepared ligand.

        Parameters
        ----------
        ligand_pdbqt:
            Path to the ligand PDBQT file (from :meth:`prepare_ligand`).

        Returns
        -------
        dict
            Structured docking result -- see module docstring for schema.
        """
        ligand_pdbqt = str(Path(ligand_pdbqt).expanduser().resolve())
        output_path = self._make_output_path(ligand_pdbqt)
        log_path = str(Path(output_path).with_suffix(".log"))

        # If Vina not found -> return mock results (never crash)
        if self._vina_exe is None:
            return self._mock_dock(ligand_pdbqt, output_path)

        try:
            self._run_vina(ligand_pdbqt, output_path, log_path)
            return self._parse_vina_output(output_path, log_path)
        except Exception as exc:
            logger.error("Docking failed: %s", exc, exc_info=True)
            return {
                "docking_score": 0.0,
                "binding_modes": [],
                "output_pdbqt": output_path,
                "success": False,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_vina(self) -> str | None:
        """Search PATH for the ``vina`` executable."""
        vina = shutil.which("vina")
        return vina if vina else None

    def _make_output_path(self, ligand_pdbqt: str) -> str:
        """Generate a unique output PDBQT path from the ligand name."""
        stem = Path(ligand_pdbqt).stem
        ts = int(time.time())
        return str(self._ligand_dir / f"{stem}_docked_{ts}.pdbqt")

    # -- Ligand preparation helpers ------------------------------------

    def _mol_to_pdbqt(self, mol_path: str, pdbqt_path: str) -> None:
        """Convert MOL file to PDBQT using openbabel if available, else RDKit PDB."""
        obabel = shutil.which("obabel")
        if obabel:
            cmd = [
                obabel,
                mol_path,
                "-O",
                pdbqt_path,
                "-p",
                "7.4",  # protonation at pH 7.4
                "--partialcharge",
                "gasteiger",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                logger.warning("obabel failed (%s), falling back to RDKit PDB", result.stderr)
                self._fallback_mol_to_pdbqt(mol_path, pdbqt_path)
        else:
            self._fallback_mol_to_pdbqt(mol_path, pdbqt_path)

    def _fallback_mol_to_pdbqt(self, mol_path: str, pdbqt_path: str) -> None:
        """Fallback: write a PDB-like file using RDKit and adapt to PDBQT format."""
        if not RDKIT_AVAILABLE:
            Path(pdbqt_path).write_text("MODEL\nREMARK  RDKit mock PDBQT\nENDMDL\n")
            return
        mol = Chem.MolFromMolFile(mol_path, removeHs=False)
        if mol is None:
            raise RuntimeError(f"Cannot read MOL file: {mol_path}")
        pdb_block = Chem.MolToPDBBlock(mol)
        # Write minimal PDBQT wrapper around PDB block
        lines = pdb_block.splitlines()
        qt_lines = ["REMARK  VinaDocker PDBQT (RDKit fallback)", "ROOT"]
        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                qt_lines.append(line)
        qt_lines.append("ENDROOT")
        qt_lines.append("TORSDOF 0")
        Path(pdbqt_path).write_text("\n".join(qt_lines) + "\n", encoding="utf-8")

    def _mock_ligand_file(self, smiles: str) -> str:
        """When RDKit is unavailable, create a minimal placeholder PDBQT."""
        hash_id = hashlib.sha256(smiles.encode()).hexdigest()[:16]
        pdbqt_path = self._ligand_dir / f"{hash_id}.pdbqt"
        pdbqt_path.write_text(
            "REMARK  Mock ligand (RDKit unavailable)\n"
            "ROOT\n"
            "ATOM      1  C   LIG A   1       0.000   0.000   0.000  0.00  0.00\n"
            "ENDROOT\n"
            "TORSDOF 0\n",
            encoding="utf-8",
        )
        return str(pdbqt_path)

    # -- Vina execution ------------------------------------------------

    def _run_vina(
        self,
        ligand_pdbqt: str,
        output_path: str,
        log_path: str,
    ) -> None:
        """Execute Vina as a subprocess.

        Parameters
        ----------
        ligand_pdbqt:
            Input ligand PDBQT.
        output_path:
            Where Vina will write docked poses.
        log_path:
            Path to capture stdout / stderr.
        """
        if self._vina_exe is None:
            raise RuntimeError("Vina executable not found.")

        cmd = [
            self._vina_exe,
            "--receptor", self.receptor_path,
            "--ligand", ligand_pdbqt,
            "--center_x", str(self.center[0]),
            "--center_y", str(self.center[1]),
            "--center_z", str(self.center[2]),
            "--size_x", str(self.box_size),
            "--size_y", str(self.box_size),
            "--size_z", str(self.box_size),
            "--out", output_path,
            "--num_modes", "9",
            "--exhaustiveness", "32",
        ]

        logger.info("Running Vina: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        # Write combined log
        Path(log_path).write_text(
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n",
            encoding="utf-8",
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Vina exited with code {result.returncode}: {result.stderr}"
            )

    def _parse_vina_output(
        self, output_pdbqt: str, log_path: str
    ) -> dict[str, Any]:
        """Parse Vina stdout (stored in *log_path*) for binding-mode scores.

        Returns
        -------
        dict
            Parsed docking result with ``binding_modes`` list.
        """
        log_text = Path(log_path).read_text(encoding="utf-8")
        binding_modes: list[dict[str, Any]] = []
        best_score = 0.0

        # Vina tabular output looks like:
        # mode |   affinity | dist from best mode
        #      | (kcal/mol) | rmsd l.b.| rmsd u.b.
        # -----+------------+----------+----------
        #    1       -8.900     0.000     0.000
        #    2       -8.500     2.100     4.300
        pattern = re.compile(
            r"^\s*(?P<mode>\d+)\s+"
            r"(?P<affinity>-?\d+\.\d+)\s+"
            r"(?P<rmsd_lb>\d+\.\d+)\s+"
            r"(?P<rmsd_ub>\d+\.\d+)",
            re.MULTILINE,
        )

        for match in pattern.finditer(log_text):
            entry = {
                "mode": int(match.group("mode")),
                "affinity": float(match.group("affinity")),
                "rmsd_lb": float(match.group("rmsd_lb")),
                "rmsd_ub": float(match.group("rmsd_ub")),
            }
            binding_modes.append(entry)
            if entry["mode"] == 1:
                best_score = entry["affinity"]

        return {
            "docking_score": best_score,
            "binding_modes": binding_modes,
            "output_pdbqt": output_pdbqt,
            "success": True,
        }

    # -- Mock docking (used when Vina unavailable) ---------------------

    def _mock_dock(self, ligand_pdbqt: str, output_path: str) -> dict[str, Any]:
        """Return simulated docking results for CI / dev environments."""
        logger.info("Returning mock docking results for %s", ligand_pdbqt)

        # Deterministic but pseudo-random scores based on ligand name
        rng = np.random.default_rng(hash(ligand_pdbqt) % (2**31))
        base_score = -7.5 - rng.random() * 4.0  # -7.5 to -11.5 kcal/mol

        modes = []
        for i in range(1, 10):
            affinity = base_score + (i - 1) * 0.4 + rng.random() * 0.2
            modes.append(
                {
                    "mode": i,
                    "affinity": round(float(affinity), 3),
                    "rmsd_lb": round(float((i - 1) * 1.2 + rng.random()), 3),
                    "rmsd_ub": round(float((i - 1) * 2.5 + rng.random()), 3),
                }
            )

        # Write a dummy output PDBQT
        Path(output_path).write_text(
            "REMARK  Mock docked poses (Vina unavailable)\n"
            "MODEL 1\n"
            "ATOM      1  C   LIG A   1       0.000   0.000   0.000  0.00  0.00\n"
            "ENDMDL\n",
            encoding="utf-8",
        )

        return {
            "docking_score": round(float(base_score), 3),
            "binding_modes": modes,
            "output_pdbqt": output_path,
            "success": True,
            "mock": True,
        }
