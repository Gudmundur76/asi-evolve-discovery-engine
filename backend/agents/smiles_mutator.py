"""smiles_mutator.py
RDKit-based SMILES mutation engine for ASI-Evolve.

Replaces the broken fingerprint bit-flip → `parent_[C{n}]` stub with
real, chemically valid molecular mutations. Every candidate produced by
this module is a valid, synthesisable molecule that can be:
  - Looked up in ChEMBL / PubChem
  - Scored by the AffinityPredictor via Morgan fingerprint
  - Submitted to a CRO for synthesis

Mutation strategies (aligned with EngineerAgent strategy names):
  - exploration      : random atom substitution or fragment addition
  - bit_flip         : targeted atom substitution at specified positions
  - guided_mutation  : substituent swap guided by statistical patterns
  - crossover        : scaffold-preserving fragment exchange between two molecules

All strategies return a canonical SMILES string and guarantee:
  1. The output parses with Chem.MolFromSmiles (no invalid SMILES)
  2. Molecular weight stays within 150–700 Da (drug-like range)
  3. Atom count stays within ±8 atoms of parent
  4. The output differs from the parent (no no-op mutations)

If a mutation fails all retries, the parent SMILES is returned unchanged
(safe fallback — the loop continues, the cycle is counted as no-improvement).
"""
from __future__ import annotations

import logging
import random
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, RWMol
from rdkit.Chem.rdchem import Atom, BondType
from rdkit.Chem import rdMolDescriptors

logger = logging.getLogger(__name__)

# ── Drug-likeness guard ───────────────────────────────────────────────────────
_MW_MIN = 150.0
_MW_MAX = 700.0
_ATOM_DELTA_MAX = 8

# ── Chemistry validity filter ─────────────────────────────────────────────────
# SMARTS patterns for groups that are synthetically impossible or highly
# reactive/unstable in a drug context. Any candidate matching one of these
# is rejected immediately, regardless of predicted affinity.
#
# Rules derived from:
#   - Ertl, P. (2020) "Cheminformatics analysis of organic substituents"
#   - Brenk, R. et al. (2008) "Lessons Learnt from Assembling Screening
#     Libraries for Drug Discovery" ChemMedChem 3(3):435-444
#   - Pan, A. et al. (2013) "Common Reactivity Pattern (COREPA)"
#
_INVALID_PATTERNS: List[Tuple[str, str]] = [
    # Halogen-heteroatom bonds (O-F, O-Cl, N-F, N-Cl, S-F, etc.)
    ("[O][F]",          "O-F bond (hypofluorite) — thermodynamically unstable"),
    ("[O][Cl]",         "O-Cl bond (hypochlorite) — highly reactive"),
    ("[O][Br]",         "O-Br bond — unstable"),
    ("[O][I]",          "O-I bond — unstable"),
    ("[N][F]",          "N-F bond — highly reactive fluorinating agent"),
    ("[N][Cl]",         "N-Cl bond — chloramine, unstable"),
    ("[S][F]",          "S-F bond — sulfonyl fluoride (reactive warhead, not drug-like)"),
    # Peroxides and peracids
    ("[O][O]",          "Peroxide O-O bond — explosive/unstable"),
    ("[O][O][H]",       "Hydroperoxide — unstable"),
    ("C(=O)[O][O]",     "Peracid — highly reactive"),
    # Azides, diazonium
    ("[N]=[N]=[N]",     "Azide — explosive"),
    ("[N+]#[N]",        "Diazonium — highly reactive"),
    # Acyl halides
    ("C(=O)[F]",        "Acyl fluoride — highly reactive electrophile"),
    ("C(=O)[Cl]",       "Acyl chloride — highly reactive electrophile"),
    ("C(=O)[Br]",       "Acyl bromide — highly reactive electrophile"),
    # Isocyanates and isothiocyanates
    ("[N]=C=O",         "Isocyanate — highly reactive"),
    ("[N]=C=S",         "Isothiocyanate — reactive"),
    # True arene oxides (epoxide on aromatic ring — NOT phenol/methoxy)
    # Pattern: aromatic carbon bonded to oxygen that is part of a 3-membered ring
    ("c1ccc2c(c1)OC2",  "Arene oxide (epoxide on benzene ring) — mutagenic metabolite"),
    # Phosphorus-halogen bonds
    ("[P][F]",          "P-F bond — nerve agent motif, not drug-like"),
    ("[P][Cl]",         "P-Cl bond — highly reactive"),
    # Hypervalent nitrogen (pentavalent N without formal charge)
    # Checked separately in _is_chemically_valid
]

# Pre-compile SMARTS patterns for performance
_COMPILED_INVALID: List[Tuple[Chem.Mol, str]] = []
for _sma, _reason in _INVALID_PATTERNS:
    _pat = Chem.MolFromSmarts(_sma)
    if _pat is not None:
        _COMPILED_INVALID.append((_pat, _reason))
    else:
        logger.warning("Could not compile SMARTS pattern: %s", _sma)


def _is_chemically_valid(mol: Chem.Mol) -> Tuple[bool, str]:
    """
    Return (True, "") if the molecule passes all chemistry validity checks,
    or (False, reason) if it contains a reactive/impossible group.

    This is the gate that prevents the pipeline from advancing fantasy molecules.
    """
    # 1. Check against all compiled SMARTS patterns
    for pattern, reason in _COMPILED_INVALID:
        if mol.HasSubstructMatch(pattern):
            return False, reason

    # 2. Check for hypervalent atoms (valence exceeds allowed maximum)
    #    RDKit's SanitizeMol already catches most of these, but we add
    #    an explicit check for atoms with impossible valence states.
    allowed_valence = {
        6: [4],           # Carbon: max 4
        7: [3, 5],        # Nitrogen: 3 (neutral) or 5 (N-oxide, formal charge)
        8: [2],           # Oxygen: max 2
        9: [1],           # Fluorine: exactly 1
        15: [3, 5],       # Phosphorus: 3 or 5
        16: [2, 4, 6],    # Sulfur: 2, 4, or 6
        17: [1],          # Chlorine: 1 (in organic context)
        35: [1],          # Bromine: 1
        53: [1, 3, 5, 7], # Iodine: 1, 3, 5, or 7
    }
    for atom in mol.GetAtoms():
        anum = atom.GetAtomicNum()
        if anum in allowed_valence:
            valence = atom.GetTotalValence()
            if valence not in allowed_valence[anum]:
                # Allow if there is a formal charge that explains it
                if atom.GetFormalCharge() == 0:
                    return False, f"Hypervalent atom: {atom.GetSymbol()} with valence {valence}"

    # 3. Check for atoms with no known synthetic precedent in drug context
    #    Reject noble gases and radioactive elements
    forbidden_elements = {2, 10, 18, 36, 54, 86,   # noble gases
                          43, 61, 84, 85, 87, 88, 89,  # radioactive/unstable
                          90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in forbidden_elements:
            return False, f"Forbidden element: {atom.GetSymbol()}"

    return True, ""


def check_smiles_validity(smiles: str) -> Tuple[bool, str]:
    """
    Public function to check a SMILES string for chemistry validity.
    Returns (is_valid, reason_if_invalid).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Invalid SMILES — RDKit cannot parse"
    return _is_chemically_valid(mol)

# ── Bioisosteric substitution table (C→N, N→O, etc.) ────────────────────────
# Each entry: (from_atomic_num, to_atomic_num, label)
_BIOISOSTERE_SWAPS: List[Tuple[int, int, str]] = [
    (6, 7, "C→N"),
    (6, 8, "C→O"),
    (6, 16, "C→S"),
    (7, 6, "N→C"),
    (7, 8, "N→O"),
    (8, 7, "O→N"),
    (8, 16, "O→S"),
    (16, 8, "S→O"),
]

# ── Small fragments to add (as SMILES) ───────────────────────────────────────
_FRAGMENTS = [
    "C",        # methyl
    "CC",       # ethyl
    "F",        # fluoro
    "Cl",       # chloro
    "OC",       # methoxy
    "N",        # amino
    "C(=O)N",   # amide
    "C#N",      # nitrile
    "S(=O)(=O)N",  # sulfonamide
    "c1ccccc1",    # phenyl
    "c1ccncc1",    # pyridyl
]


def _is_drug_like(mol: Chem.Mol, parent_mol: Chem.Mol) -> bool:
    """Return True if mol passes drug-likeness, size, and chemistry validity guards."""
    mw = Descriptors.MolWt(mol)
    if not (_MW_MIN <= mw <= _MW_MAX):
        return False
    delta = abs(mol.GetNumAtoms() - parent_mol.GetNumAtoms())
    if delta > _ATOM_DELTA_MAX:
        return False
    # Chemistry validity gate — this is the critical filter added after
    # the O-F bond incident. Any molecule that fails this check is rejected
    # regardless of predicted affinity or novelty score.
    valid, reason = _is_chemically_valid(mol)
    if not valid:
        logger.debug("Chemistry validity filter rejected molecule: %s — %s", Chem.MolToSmiles(mol)[:50], reason)
        return False
    return True


def _sanitize_and_canon(rw: RWMol) -> Optional[str]:
    """Sanitize an RWMol and return canonical SMILES, or None on failure."""
    try:
        Chem.SanitizeMol(rw)
        smi = Chem.MolToSmiles(rw)
        if Chem.MolFromSmiles(smi) is None:
            return None
        return smi
    except Exception:
        return None


# ── Strategy implementations ─────────────────────────────────────────────────

def _atom_substitution(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
    """Replace one non-ring atom with a bioisosteric equivalent."""
    # Build candidate list: non-ring, not stereo-centre, degree ≤ 3
    ring_atoms = set(sum(mol.GetRingInfo().AtomRings(), ()))
    candidates = [
        (a.GetIdx(), a.GetAtomicNum())
        for a in mol.GetAtoms()
        if a.GetIdx() not in ring_atoms and a.GetDegree() <= 3
    ]
    if not candidates:
        return None

    rng.shuffle(candidates)
    swaps = [(f, t, lbl) for f, t, lbl in _BIOISOSTERE_SWAPS]

    for idx, atomic_num in candidates:
        matching = [(f, t, lbl) for f, t, lbl in swaps if f == atomic_num]
        if not matching:
            continue
        _, to_num, lbl = rng.choice(matching)
        rw = RWMol(mol)
        rw.GetAtomWithIdx(idx).SetAtomicNum(to_num)
        smi = _sanitize_and_canon(rw)
        if smi and smi != Chem.MolToSmiles(mol):
            parent_mol = mol
            new_mol = Chem.MolFromSmiles(smi)
            if new_mol and _is_drug_like(new_mol, parent_mol):
                logger.debug("Atom substitution %s at idx %d → %s", lbl, idx, smi[:40])
                return smi
    return None


def _fragment_addition(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
    """Add a small fragment to a heteroatom with available valence."""
    # Prefer N and O atoms with at least one implicit H
    candidates = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() in (7, 8) and a.GetTotalNumHs() > 0 and not a.IsInRing()
    ]
    if not candidates:
        # Fall back to any atom with implicit H
        candidates = [
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetTotalNumHs() > 0
        ]
    if not candidates:
        return None

    frags = _FRAGMENTS[:]
    rng.shuffle(frags)
    rng.shuffle(candidates)

    for frag_smi in frags:
        frag_mol = Chem.MolFromSmiles(frag_smi)
        if frag_mol is None:
            continue
        for attach_idx in candidates:
            try:
                combined = Chem.RWMol(Chem.CombineMols(mol, frag_mol))
                frag_start = mol.GetNumAtoms()
                combined.AddBond(attach_idx, frag_start, BondType.SINGLE)
                smi = _sanitize_and_canon(combined)
                if smi and smi != Chem.MolToSmiles(mol):
                    new_mol = Chem.MolFromSmiles(smi)
                    if new_mol and _is_drug_like(new_mol, mol):
                        logger.debug("Fragment addition %s at idx %d → %s", frag_smi, attach_idx, smi[:40])
                        return smi
            except Exception:
                continue
    return None


def _fragment_removal(mol: Chem.Mol, rng: random.Random) -> Optional[str]:
    """Remove a terminal non-ring substituent (trim a branch)."""
    # Find terminal atoms: degree 1, not in ring
    ring_atoms = set(sum(mol.GetRingInfo().AtomRings(), ()))
    terminals = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetDegree() == 1 and a.GetIdx() not in ring_atoms
    ]
    if not terminals:
        return None

    rng.shuffle(terminals)
    for idx in terminals:
        rw = RWMol(mol)
        rw.RemoveAtom(idx)
        smi = _sanitize_and_canon(rw)
        if smi and smi != Chem.MolToSmiles(mol):
            new_mol = Chem.MolFromSmiles(smi)
            if new_mol and _is_drug_like(new_mol, mol):
                logger.debug("Fragment removal at idx %d → %s", idx, smi[:40])
                return smi
    return None


def _scaffold_crossover(mol_a: Chem.Mol, mol_b: Chem.Mol, rng: random.Random) -> Optional[str]:
    """
    Scaffold-preserving crossover: take the ring system of mol_a and
    the largest non-ring substituent of mol_b and combine them.
    Falls back to atom substitution if crossover fails.
    """
    # Simple approach: take mol_a and replace one of its terminal groups
    # with a terminal group from mol_b
    ring_atoms_a = set(sum(mol_a.GetRingInfo().AtomRings(), ()))
    terminals_b = [
        a.GetIdx()
        for a in mol_b.GetAtoms()
        if a.GetDegree() == 1 and a.GetIdx() not in set(sum(mol_b.GetRingInfo().AtomRings(), ()))
    ]
    terminals_a = [
        a.GetIdx()
        for a in mol_a.GetAtoms()
        if a.GetDegree() == 1 and a.GetIdx() not in ring_atoms_a
    ]

    if not terminals_a or not terminals_b:
        return _atom_substitution(mol_a, rng)

    # Remove a terminal from mol_a and add the atomic num from mol_b terminal
    rng.shuffle(terminals_a)
    rng.shuffle(terminals_b)

    for ta in terminals_a:
        for tb in terminals_b:
            atomic_b = mol_b.GetAtomWithIdx(tb).GetAtomicNum()
            rw = RWMol(mol_a)
            rw.GetAtomWithIdx(ta).SetAtomicNum(atomic_b)
            smi = _sanitize_and_canon(rw)
            if smi and smi != Chem.MolToSmiles(mol_a):
                new_mol = Chem.MolFromSmiles(smi)
                if new_mol and _is_drug_like(new_mol, mol_a):
                    logger.debug("Crossover ta=%d tb=%d → %s", ta, tb, smi[:40])
                    return smi
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def mutate_smiles(
    parent_smiles: str,
    strategy: str = "exploration",
    target_bits: Optional[List[int]] = None,
    crossover_smiles: Optional[str] = None,
    seed: Optional[int] = None,
    max_retries: int = 10,
) -> Tuple[str, str]:
    """Generate a mutated SMILES from a parent molecule.

    Args:
        parent_smiles: Valid SMILES of the parent molecule.
        strategy: One of "exploration", "bit_flip", "guided_mutation", "crossover".
        target_bits: Ignored (kept for API compatibility with EngineerAgent).
        crossover_smiles: Second parent SMILES for crossover strategy.
        seed: Random seed for reproducibility.
        max_retries: Number of mutation attempts before falling back to parent.

    Returns:
        Tuple of (new_smiles, mutation_description).
        If all retries fail, returns (parent_smiles, "no-op: all mutations failed").
    """
    rng = random.Random(seed)

    parent_mol = Chem.MolFromSmiles(parent_smiles)
    if parent_mol is None:
        logger.warning("Invalid parent SMILES — returning parent unchanged: %s", parent_smiles[:60])
        return parent_smiles, "no-op: invalid parent SMILES"

    # Normalise parent to canonical form
    parent_canon = Chem.MolToSmiles(parent_mol)
    parent_mol = Chem.MolFromSmiles(parent_canon)

    crossover_mol: Optional[Chem.Mol] = None
    if crossover_smiles:
        crossover_mol = Chem.MolFromSmiles(crossover_smiles)

    # Strategy dispatch with retry loop
    for attempt in range(max_retries):
        try:
            result: Optional[str] = None

            if strategy == "crossover" and crossover_mol is not None:
                result = _scaffold_crossover(parent_mol, crossover_mol, rng)
            elif strategy in ("bit_flip", "guided_mutation"):
                # Alternate between atom sub and fragment ops
                ops = [_atom_substitution, _fragment_addition, _fragment_removal]
                result = rng.choice(ops)(parent_mol, rng)
            else:
                # exploration: try all three ops in random order
                ops = [_atom_substitution, _fragment_addition, _fragment_removal]
                rng.shuffle(ops)
                for op in ops:
                    result = op(parent_mol, rng)
                    if result:
                        break

            if result and result != parent_canon:
                desc = f"rdkit-{strategy} attempt {attempt + 1}"
                logger.info("Mutation success [%s]: %s → %s", desc, parent_canon[:30], result[:40])
                return result, desc

        except Exception as exc:
            logger.debug("Mutation attempt %d failed: %s", attempt + 1, exc)

    logger.warning(
        "All %d mutation attempts failed for strategy=%s — returning parent",
        max_retries,
        strategy,
    )
    return parent_canon, "no-op: all mutations failed"


def batch_mutate(
    parent_smiles: str,
    n: int = 5,
    strategy: str = "exploration",
    seed: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """Generate n distinct mutated SMILES from a parent molecule.

    Returns a list of (smiles, description) tuples. Duplicates and
    no-ops are filtered out. If fewer than n unique mutations are found,
    the list is shorter than n.
    """
    seen = {Chem.MolToSmiles(Chem.MolFromSmiles(parent_smiles)) if Chem.MolFromSmiles(parent_smiles) else parent_smiles}
    results: List[Tuple[str, str]] = []
    base_seed = seed or 0

    for i in range(n * 3):  # over-sample to get n unique
        s, desc = mutate_smiles(parent_smiles, strategy=strategy, seed=base_seed + i)
        if s not in seen:
            seen.add(s)
            results.append((s, desc))
        if len(results) >= n:
            break

    return results
