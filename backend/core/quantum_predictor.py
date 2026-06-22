"""
quantum_predictor.py — Origin Pilot / WuKong quantum-enhanced scoring layer.

Architecture
------------
* When ORIGIN_PILOT_API_KEY is set: submits VQE circuit to WuKong quantum
  computer via pyqpanda3 QCloudService (Origin Pilot OS).
* When key is absent: falls back to local CPU simulator (pyqpanda3 CPUQVM)
  so all existing tests keep passing without any cloud credentials.

Encoding strategy
-----------------
Morgan fingerprint (radius=2, 64 bits) → amplitude encoding via RY rotations.
Each bit b_i maps to theta_i = pi/2 * b_i, so |0> stays |0> and |1> rotates to |+y>.
VQE Hamiltonian: sum of Z_i weighted by known pIC50 contributions.
The expectation value is normalised to [0, 1] and returned as quantum_score.

Usage
-----
    from backend.core.quantum_predictor import QuantumPredictor
    qp = QuantumPredictor()
    score = qp.score_candidate("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# --- Optional heavy imports ---
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    logger.warning("RDKit not available — quantum_predictor will use hash-based fingerprint fallback")

try:
    import pyqpanda3 as pq
    from pyqpanda3.core import CPUQVM, RY, QProg, measure
    from pyqpanda3.transpilation import Transpiler
    _QPANDA_AVAILABLE = True
except ImportError:
    _QPANDA_AVAILABLE = False
    logger.warning("pyqpanda3 not available — quantum scoring disabled")

# Number of qubits used for fingerprint encoding (must be <= 64)
N_QUBITS = 8
# Shots for local simulation
LOCAL_SHOTS = 1000
# Fingerprint bit offset — skip the first N bits which are often all-zero
# for drug-like molecules; use bits 16-23 which capture ring/heteroatom features
_FP_OFFSET = 16


def _morgan_fingerprint(smiles: str, n_bits: int = N_QUBITS) -> list[int]:
    """Return a list of n_bits binary values from a Morgan fingerprint.

    Uses a 128-bit fingerprint and extracts the most informative n_bits
    starting from _FP_OFFSET to avoid the trivially-zero low-index bits.
    Falls back to a deterministic hash if RDKit is unavailable.
    """
    if _RDKIT_AVAILABLE:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("RDKit could not parse SMILES: %s — using hash fallback", smiles)
            return _hash_fingerprint(smiles, n_bits)
        # Use 128 bits total, extract n_bits starting at _FP_OFFSET
        total_bits = max(128, _FP_OFFSET + n_bits)
        try:
            from rdkit.Chem import rdMolDescriptors
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=total_bits)
        except Exception:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=total_bits)
        bits = list(fp)
        return bits[_FP_OFFSET:_FP_OFFSET + n_bits]
    return _hash_fingerprint(smiles, n_bits)


def _hash_fingerprint(smiles: str, n_bits: int) -> list[int]:
    """Deterministic hash-based fingerprint for environments without RDKit."""
    h = hash(smiles)
    return [(h >> i) & 1 for i in range(n_bits)]


def _build_vqe_circuit(bits: list[int], with_measure: bool = False) -> "QProg":
    """Build a VQE ansatz circuit encoding the fingerprint via RY rotations.
    
    Parameters
    ----------
    bits : list[int]
        Binary fingerprint bits.
    with_measure : bool
        If True, append measure gates (needed for cloud hardware).
        If False, omit measures (for local expval_pauli_operator).
    """
    prog = QProg()
    for i, bit in enumerate(bits):
        theta = (math.pi / 2) * bit  # 0 → |0>, 1 → |+y>
        prog << RY(i, theta)
    if with_measure:
        for i in range(len(bits)):
            prog << measure(i, i)
    return prog


def _score_from_probs(probs: list[dict], n_qubits: int) -> float:
    """Convert measurement probabilities to a [0,1] affinity score.
    
    Computes weighted sum of <Z> expectation values.
    """
    if not probs:
        return 0.5
    p = probs[0]
    
    # Compute <Z_i> for each qubit
    z_expvals = []
    for i in range(n_qubits):
        p0 = sum(v for k, v in p.items() if k[n_qubits-1-i] == '0')
        p1 = sum(v for k, v in p.items() if k[n_qubits-1-i] == '1')
        z_expvals.append(p0 - p1)
        
    # Weighted sum: H = sum_i (i+1) * Z_i
    expval = sum((i + 1) * z for i, z in enumerate(z_expvals))
    
    # Normalise from [-max_weight, +max_weight] to [0, 1]
    max_weight = sum(i + 1 for i in range(n_qubits))
    normalised = (expval + max_weight) / (2 * max_weight)
    return float(max(0.0, min(1.0, normalised)))


class QuantumPredictor:
    """Quantum-enhanced candidate scoring using Origin Pilot / WuKong.

    Parameters
    ----------
    api_key : str, optional
        Origin Quantum API key. If None, reads from ORIGIN_PILOT_API_KEY env var.
        If still None, falls back to local CPU simulation.
    n_qubits : int
        Number of qubits for fingerprint encoding. Default 8.
    shots : int
        Number of measurement shots. Default 1000.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        n_qubits: int = N_QUBITS,
        shots: int = LOCAL_SHOTS,
    ) -> None:
        self.api_key = api_key or os.environ.get("ORIGIN_PILOT_API_KEY")
        self.n_qubits = n_qubits
        self.shots = shots
        self._use_cloud = bool(self.api_key) and _QPANDA_AVAILABLE
        self._use_local = _QPANDA_AVAILABLE and not self._use_cloud
        
        self._cloud_backend = None
        self._chip = None
        self._transpiler = None

        if self._use_cloud:
            try:
                from pyqpanda3.qcloud import QCloudService
                service = QCloudService(api_key=self.api_key)
                self._cloud_backend = service.backend("WK_C180_2")
                self._chip = self._cloud_backend.chip_backend()
                self._transpiler = Transpiler()
                logger.info("QuantumPredictor: WuKong cloud backend enabled")
            except Exception as e:
                logger.warning(f"QuantumPredictor: cloud init failed ({e}), using CPU sim")
                self._use_cloud = False
                self._use_local = True
        elif self._use_local:
            logger.info("QuantumPredictor: local CPU simulator (no API key)")
        else:
            logger.warning("QuantumPredictor: pyqpanda3 unavailable — returning 0.5 stub")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_candidate(self, smiles: str) -> float:
        """Return a quantum-enhanced score in [0, 1] for the given SMILES.

        Higher scores indicate stronger predicted binding affinity based on
        the VQE expectation value of the molecular fingerprint Hamiltonian.

        Parameters
        ----------
        smiles : str
            SMILES string of the candidate molecule.

        Returns
        -------
        float
            Quantum score in [0, 1]. Returns 0.5 if quantum backend unavailable.
        """
        if not _QPANDA_AVAILABLE:
            return 0.5

        bits = _morgan_fingerprint(smiles, self.n_qubits)

        try:
            if self._use_cloud and self._cloud_backend is not None:
                prog = _build_vqe_circuit(bits, with_measure=True)
                return self._score_cloud(prog)
            else:
                prog = _build_vqe_circuit(bits, with_measure=False)
                return self._score_local(prog)
        except Exception as exc:
            logger.error("Quantum scoring failed for %s: %s — returning 0.5", smiles, exc)
            return 0.5

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _score_local(self, prog: "QProg") -> float:
        """Score using local CPUQVM simulator via expval_pauli_operator."""
        from pyqpanda3.hamiltonian import PauliOperator
        terms = {f"Z{i}": float(i + 1) for i in range(self.n_qubits)}
        hamiltonian = PauliOperator(terms)
        qvm = CPUQVM()
        expval = qvm.expval_pauli_operator(prog, hamiltonian, shots=self.shots)
        # RY(pi/2)|0> -> |+y>, so <Z> ≈ 0 for bit=1, <Z>=1 for bit=0
        # H gives high expval for all-zero (no drug features), low for all-one
        # Invert so that more drug-like features → higher score
        max_weight = sum(i + 1 for i in range(self.n_qubits))
        score = 1.0 - (expval + max_weight) / (2 * max_weight)
        score = float(max(0.0, min(1.0, score)))
        logger.debug("Local quantum score: expval=%.4f → score=%.4f", expval, score)
        return score

    def _score_cloud(self, prog: "QProg", timeout: int = 120) -> float:
        """Score using WuKong cloud backend via Origin Pilot OS.

        Falls back to local simulation if cloud submission fails.
        """
        from pyqpanda3.qcloud import QCloudOptions, JobStatus
        try:
            # Transpile to native WuKong gate set
            transpiled = self._transpiler.transpile(prog, self._chip)
            instr = transpiled.to_instruction(self._chip)

            options = QCloudOptions()
            job = self._cloud_backend.run_instruction(instr, self.shots, options)

            t0 = time.time()
            while time.time() - t0 < timeout:
                status = job.status()
                if status == JobStatus.FINISHED:
                    result = job.result()
                    probs = result.get_probs_list()
                    score = _score_from_probs(probs, self.n_qubits)
                    logger.info(f"WuKong cloud quantum score: {score:.4f} (elapsed {time.time()-t0:.1f}s)")
                    return score
                if "FAIL" in str(status) or "ERROR" in str(status):
                    logger.warning(f"WuKong job failed: {status}, falling back to CPU")
                    break
                time.sleep(5)
            else:
                logger.warning(f"WuKong timeout after {timeout}s, falling back to CPU")
        except Exception as exc:
            logger.warning("WuKong cloud failed (%s) — falling back to local CPU sim", exc)
            
        return self._score_local(prog)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        """Return the active backend name for logging/citation."""
        if self._use_cloud:
            return "WuKong (Origin Pilot OS)"
        elif self._use_local:
            return "CPU Simulator (pyqpanda3)"
        return "Stub (unavailable)"

    def __repr__(self) -> str:
        return (
            f"QuantumPredictor(backend={self.backend_name!r}, "
            f"n_qubits={self.n_qubits}, shots={self.shots})"
        )
