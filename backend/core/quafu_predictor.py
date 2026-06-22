"""
Quafu (BAQIS) Quantum Predictor for HIV-1 Protease Candidate Scoring
=====================================================================
Uses the Quafu quantum cloud platform (quafu.baqis.ac.cn) operated by the
Beijing Academy of Quantum Information Sciences (BAQIS) to run VQE-style
circuits on superconducting ScQ quantum hardware.

Activation:
    1. Register at https://quafu.baqis.ac.cn
    2. Go to Account Settings → API Token → Copy token
    3. Set env var: QUAFU_API_KEY=<your_token>
    4. Add as GitHub secret: QUAFU_API_KEY=<your_token>

When QUAFU_API_KEY is not set, falls back to local CPU simulation using
the quafu simulator backend (no hardware access required).

Backend selection (auto-detected from available ScQ backends):
    - ScQ-P10  : 10-qubit superconducting processor
    - ScQ-P20  : 20-qubit superconducting processor
    - ScQ-P50  : 50-qubit superconducting processor (preferred)
    - simulator: Local quafu simulator (fallback)
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — quafu may not be installed in all environments
# ---------------------------------------------------------------------------
try:
    from quafu import QuantumCircuit, User
    from quafu.results.results import ExecResult
    _QUAFU_AVAILABLE = True
except ImportError:
    _QUAFU_AVAILABLE = False
    logger.warning("pyquafu not installed. Run: pip install pyquafu")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QUAFU_API_KEY = os.environ.get("QUAFU_API_KEY", "")
N_QUBITS = 8          # Morgan fingerprint bits to encode
N_SHOTS = 1000        # Measurement shots on real hardware
PREFERRED_BACKENDS = ["ScQ-P50", "ScQ-P20", "ScQ-P10"]  # priority order
TIMEOUT_SECONDS = 120


@dataclass
class QuafuResult:
    smiles: str
    score: float                    # 0.0 – 1.0 VQE affinity score
    backend: str                    # e.g. "ScQ-P50" or "simulator"
    elapsed_s: float
    raw_counts: dict = field(default_factory=dict)
    error: Optional[str] = None
    stub_active: bool = True        # True when no API key / not on real hardware


# ---------------------------------------------------------------------------
# Fingerprint encoding
# ---------------------------------------------------------------------------
def _smiles_to_angles(smiles: str, n_qubits: int = N_QUBITS) -> list[float]:
    """Convert SMILES to RY rotation angles via molecular descriptors."""
    if not _RDKIT_AVAILABLE:
        import hashlib
        h = int(hashlib.sha256(smiles.encode()).hexdigest(), 16)
        return [((h >> (i * 8)) & 0xFF) / 255.0 * math.pi for i in range(n_qubits)]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [math.pi / 4] * n_qubits

    from rdkit.Chem import Descriptors
    # Extract 8 distinct molecular descriptors mapped to [0, π]
    mw = min(1.0, Descriptors.MolWt(mol) / 1000.0)
    logp = max(0.0, min(1.0, (Descriptors.MolLogP(mol) + 2) / 8.0))
    tpsa = min(1.0, Descriptors.TPSA(mol) / 200.0)
    hbd = min(1.0, Descriptors.NumHDonors(mol) / 10.0)
    hba = min(1.0, Descriptors.NumHAcceptors(mol) / 15.0)
    rot = min(1.0, Descriptors.NumRotatableBonds(mol) / 20.0)
    rings = min(1.0, Descriptors.RingCount(mol) / 10.0)
    heavy = min(1.0, Descriptors.HeavyAtomCount(mol) / 50.0)

    features = [mw, logp, tpsa, hbd, hba, rot, rings, heavy]
    # Pad or truncate to n_qubits
    if len(features) < n_qubits:
        features += [0.5] * (n_qubits - len(features))
    features = features[:n_qubits]

    return [f * math.pi for f in features]


# ---------------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------------
def _build_vqe_circuit(angles: list[float], n_qubits: int = N_QUBITS) -> "QuantumCircuit":
    """
    Build a hardware-efficient VQE ansatz circuit.

    Layer 1: RY rotations encoding the molecular fingerprint
    Layer 2: CNOT entanglement chain (q0→q1→...→q_{n-1})
    Layer 3: RY rotations (variational layer, fixed at π/4 for scoring)
    Measurement: all qubits
    """
    if not _QUAFU_AVAILABLE:
        raise RuntimeError("pyquafu not installed")

    qc = QuantumCircuit(n_qubits, n_qubits)

    # Layer 1: encode fingerprint
    for i, angle in enumerate(angles):
        qc.ry(i, angle)

    # Layer 2: entanglement
    for i in range(n_qubits - 1):
        qc.cnot(i, i + 1)

    # Layer 3: variational (fixed scoring layer)
    for i in range(n_qubits):
        qc.ry(i, math.pi / 4)

    # Measure all
    qc.measure(list(range(n_qubits)), list(range(n_qubits)))

    return qc


# ---------------------------------------------------------------------------
# Score extraction from measurement counts
# ---------------------------------------------------------------------------
def _counts_to_score(counts: dict, n_qubits: int = N_QUBITS) -> float:
    """
    Convert measurement counts to a scalar affinity score in [0, 1].

    Score = weighted average of Hamming weight / n_qubits across all bitstrings.
    High Hamming weight → more qubits in |1⟩ → higher affinity proxy.
    """
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.5

    weighted_sum = 0.0
    for bitstring, count in counts.items():
        # bitstring may be int or str depending on quafu version
        if isinstance(bitstring, int):
            hw = bin(bitstring).count("1")
        else:
            hw = str(bitstring).count("1")
        weighted_sum += (hw / n_qubits) * count

    return weighted_sum / total_shots


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
def _select_backend(user: "User") -> str:
    """Select the best available ScQ backend, falling back to simulator."""
    try:
        backends = user.get_available_backends(print_info=False)
        for preferred in PREFERRED_BACKENDS:
            for b in backends:
                name = b.get("system_name", "") if isinstance(b, dict) else str(b)
                if preferred.lower() in name.lower():
                    return name
    except Exception as exc:
        logger.debug("Backend listing failed: %s", exc)
    return "simulator"


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------
class QuafuPredictor:
    """
    Quafu-based quantum affinity predictor for HIV-1 protease inhibitor candidates.

    Usage:
        predictor = QuafuPredictor()
        result = predictor.score_candidate("CC(C)CC1=CC=CC=C1")
        print(result.score, result.backend)
    """

    def __init__(
        self,
        api_key: str = QUAFU_API_KEY,
        n_qubits: int = N_QUBITS,
        n_shots: int = N_SHOTS,
        preferred_backends: list[str] = None,
    ):
        self.api_key = api_key
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.preferred_backends = preferred_backends or PREFERRED_BACKENDS
        self._user: Optional["User"] = None
        self._backend: Optional[str] = None
        self._initialized = False

    @property
    def backend_name(self) -> str:
        """Human-readable backend name."""
        self._ensure_initialized()
        if not _QUAFU_AVAILABLE:
            return "Classical Fallback (pyquafu unavailable)"
        if not self.api_key:
            return "Classical Fallback (no QUAFU_API_KEY)"
        return f"Quafu {self._backend or 'simulator'}"

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        if not _QUAFU_AVAILABLE:
            logger.warning("pyquafu unavailable — using hash-based fallback scoring")
            self._initialized = True
            return

        if not self.api_key:
            logger.info("QUAFU_API_KEY not set — using local simulator")
            self._backend = "simulator"
            self._initialized = True
            return

        try:
            self._user = User()
            self._user.save_apitoken(self.api_key)
            self._backend = _select_backend(self._user)
            logger.info("Quafu initialized — backend: %s", self._backend)
        except Exception as exc:
            logger.warning("Quafu init failed (%s) — using simulator", exc)
            self._backend = "simulator"

        self._initialized = True

    def score_candidate(self, smiles: str) -> QuafuResult:
        """Score a single SMILES string and return a QuafuResult."""
        self._ensure_initialized()
        t0 = time.time()

        angles = _smiles_to_angles(smiles, self.n_qubits)

        # --- pyquafu unavailable: hash-based deterministic fallback ---
        if not _QUAFU_AVAILABLE:
            score = sum(angles) / (self.n_qubits * math.pi)
            return QuafuResult(
                smiles=smiles,
                score=round(score, 4),
                backend="hash_fallback",
                elapsed_s=round(time.time() - t0, 2),
                stub_active=True,
            )

        # --- Local simulator path ---
        if self._backend == "simulator" or not self.api_key:
            return self._score_simulator(smiles, angles, t0)

        # --- Real hardware path ---
        return self._score_hardware(smiles, angles, t0)

    def _score_simulator(self, smiles: str, angles: list[float], t0: float) -> QuafuResult:
        """Run on local quafu simulator."""
        try:
            from quafu import simulate
            qc = _build_vqe_circuit(angles, self.n_qubits)
            # Remove measurement for statevector simulation
            qc_no_meas = QuantumCircuit(self.n_qubits)
            for i, angle in enumerate(angles):
                qc_no_meas.ry(i, angle)
            for i in range(self.n_qubits - 1):
                qc_no_meas.cnot(i, i + 1)
            for i in range(self.n_qubits):
                qc_no_meas.ry(i, math.pi / 4)

            result = simulate(qc_no_meas, output="probabilities")
            probs = result.probabilities if hasattr(result, "probabilities") else {}

            # Convert probabilities to score
            score = 0.0
            total = 0.0
            for idx, prob in enumerate(probs):
                hw = bin(idx).count("1")
                score += (hw / self.n_qubits) * prob
                total += prob
            if total > 0:
                score /= total

            return QuafuResult(
                smiles=smiles,
                score=round(score, 4),
                backend="quafu_simulator",
                elapsed_s=round(time.time() - t0, 2),
                raw_counts={},
            )
        except Exception as exc:
            logger.warning("Quafu simulator error: %s — using angle-based fallback", exc)
            score = sum(angles) / (self.n_qubits * math.pi)
            return QuafuResult(
                smiles=smiles,
                score=round(score, 4),
                backend="angle_fallback",
                elapsed_s=round(time.time() - t0, 2),
                error=str(exc),
            )

    def _score_hardware(self, smiles: str, angles: list[float], t0: float) -> QuafuResult:
        """Submit circuit to real ScQ hardware and wait for result."""
        try:
            qc = _build_vqe_circuit(angles, self.n_qubits)
            task = self._user.send_task(
                qc,
                task_name=f"asi_evolve_{hash(smiles) & 0xFFFF:04x}",
                chip_name=self._backend,
                shots=self.n_shots,
            )
            # Poll for result
            deadline = time.time() + TIMEOUT_SECONDS
            while time.time() < deadline:
                result = task.result()
                if result is not None:
                    counts = result.counts if hasattr(result, "counts") else {}
                    score = _counts_to_score(counts, self.n_qubits)
                    logger.info(
                        "Quafu %s score: %.4f (elapsed %.1fs)",
                        self._backend, score, time.time() - t0
                    )
                    return QuafuResult(
                        smiles=smiles,
                        score=round(score, 4),
                        backend=self._backend,
                        elapsed_s=round(time.time() - t0, 2),
                        raw_counts=dict(counts),
                    )
                time.sleep(5)

            raise TimeoutError(f"Quafu job timed out after {TIMEOUT_SECONDS}s")

        except Exception as exc:
            logger.warning("Quafu hardware error: %s — falling back to simulator", exc)
            return self._score_simulator(smiles, angles, t0)

    def score_batch(self, smiles_list: list[str]) -> list[QuafuResult]:
        """Score a list of SMILES strings sequentially."""
        return [self.score_candidate(s) for s in smiles_list]


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def quafu_score(smiles: str, api_key: str = QUAFU_API_KEY) -> float:
    """Quick single-molecule score. Returns float in [0, 1]."""
    predictor = QuafuPredictor(api_key=api_key)
    result = predictor.score_candidate(smiles)
    return result.score


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    test_smiles = [
        "CC(C)CC1=CC=CC=C1",                          # Ibuprofen-like
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(CN3CCN(CC3)C)C=C2)NC3=NC=CC(=N3)C4=CN=CC=C4",  # Imatinib
        "CC(C)(C)NCC(O)C1=CC(CO)=C(O)C=C1",           # Salbutamol
    ]

    predictor = QuafuPredictor()
    for smi in test_smiles:
        result = predictor.score_candidate(smi)
        print(f"SMILES: {smi[:40]}...")
        print(f"  Score:   {result.score:.4f}")
        print(f"  Backend: {result.backend}")
        print(f"  Elapsed: {result.elapsed_s:.2f}s")
        if result.error:
            print(f"  Error:   {result.error}")
        print()
