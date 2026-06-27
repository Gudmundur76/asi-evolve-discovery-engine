"""Morgan fingerprint encoding / decoding with GA-style mutations.

RDKit is imported lazily (inside methods or with a guard) so that the
module can be imported in environments where RDKit is not yet installed.
"""

import logging
import warnings
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FingerprintError(Exception):
    """Raised for fingerprint encoding/decoding failures."""

    pass


class FingerprintEncoder:
    """Encode SMILES strings into fixed-length Morgan fingerprints.

    Args:
        radius: Morgan fingerprint radius (2 = ECFP4 equivalent).
        n_bits: Length of the bit vector (default 2048).
    """

    def __init__(self, radius: int = 2, n_bits: int = 2048) -> None:
        self.radius = radius
        self.n_bits = n_bits

    # ------------------------------------------------------------------ #
    #  RDKit lazy import guard
    # ------------------------------------------------------------------ #
    @staticmethod
    def _import_rdkit():
        """Import RDKit submodules; raise FingerprintError on failure."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            return Chem, AllChem
        except ImportError as exc:
            raise FingerprintError(
                "RDKit is required for fingerprint encoding. "
                "Install it with: pip install rdkit"
            ) from exc

    # ------------------------------------------------------------------ #
    #  Core encoding
    # ------------------------------------------------------------------ #
    def smiles_to_fp(self, smiles: str) -> Optional[np.ndarray]:
        """Convert a SMILES string to a Morgan fingerprint bit vector.

        Args:
            smiles: Canonical SMILES representation of the molecule.

        Returns:
            1-D uint8 array of shape ``(n_bits,)`` with values 0/1,
            or *None* if the SMILES cannot be parsed.
        """
        Chem, AllChem = self._import_rdkit()

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug("Unable to parse SMILES: %s", smiles)
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            bitvect = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=self.radius,
                nBits=self.n_bits,
                useFeatures=False,
            )

        # Convert ExplicitBitVect to numpy array
        arr = np.zeros((self.n_bits,), dtype=np.uint8)
        # RDKit bitvects support direct indexing
        on_bits = bitvect.GetOnBits()
        arr[list(on_bits)] = 1
        return arr

    # ------------------------------------------------------------------ #
    #  Genetic-algebra helpers
    # ------------------------------------------------------------------ #
    def mutate_fp(
        self,
        fp: np.ndarray,
        num_bits: int = 3,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Return a new fingerprint with *num_bits* randomly flipped.

        Args:
            fp: Source fingerprint (1-D array of 0/1).
            num_bits: Number of positions to flip.
            seed: Optional RNG seed for reproducibility.

        Returns:
            New fingerprint array with *num_bits* bits flipped.
        """
        rng = np.random.default_rng(seed)
        mutant = fp.copy()
        if num_bits <= 0 or self.n_bits <= 0:
            return mutant
        flip_idx = rng.choice(self.n_bits, size=min(num_bits, self.n_bits), replace=False)
        flip_idx = sorted([int(i) for i in flip_idx])
        mutant[flip_idx] = 1 - mutant[flip_idx]
        return mutant

    def crossover_fps(
        self,
        fp1: np.ndarray,
        fp2: np.ndarray,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Uniform crossover between two fingerprints.

        Args:
            fp1: Parent fingerprint 1.
            fp2: Parent fingerprint 2.
            seed: Optional RNG seed.

        Returns:
            Child fingerprint array (uniform crossover of fp1 and fp2).
        """
        rng = np.random.default_rng(seed)
        mask = rng.random(self.n_bits) < 0.5
        child = np.where(mask, fp1, fp2).astype(np.uint8)
        return child

    # ------------------------------------------------------------------ #
    #  Sparse representation
    # ------------------------------------------------------------------ #
    def fp_to_sparse(self, fp: np.ndarray) -> List[int]:
        """Return the list of indices where the fingerprint is set to 1.

        Args:
            fp: Dense fingerprint array.

        Returns:
            Sorted list of integer positions of set bits.
        """
        return np.flatnonzero(fp).tolist()
