"""
recheck_candidates.py

Re-check all 9 previously generated candidates for chemically impossible
features using the new _is_chemically_valid filter. Also checks the seed
compound and the parent SMILES from the loop.
"""
import sys
sys.path.insert(0, '/home/ubuntu/repos/asi-evolve-discovery-engine')

from rdkit import Chem
from rdkit.Chem import Descriptors
from backend.agents.smiles_mutator import check_smiles_validity

# All 9 candidates from the previous run + seed
CANDIDATES = [
    # Seed / parent
    ("SEED",    "CC(C)(C)Nc1ncnc2nc(-c3ccc(OF)cc3)n(C3CC3)c12", 437.0),
    # Generated candidates
    ("A",       "CC(C)Nc1ncnc2nc(-c3ccc(OF)cc3)n(C3CC3)c12",    246.2),
    ("B",       "CC(C)(C)N(c1ncnc2nc(-c3ccc(OF)cc3)n(C3CC3)c12)S(N)(=O)=O", 206.3),
    ("C",       "CC(C)(C)Cc1ncnc2nc(-c3ccc(OF)cc3)n(C3CC3)c12", 287.2),
    ("D",       "CC(C)(C)N(C#N)c1ncnc2nc(-c3ccc(OF)cc3)n(C3CC3)c12", 353.0),
    ("E",       "CC(C)(C)N(F)c1ncnc2nc(-c3ccc(OF)cc3)n(C3CC3)c12",  395.0),
    ("F",       "CC(C)(S)Nc1ncnc2nc(-c3ccc(OF)cc3)n(C3CC3)c12",  400.0),
    ("G",       "CC(C)(C)Nc1ncnc2nc(-c3ccc(SF)cc3)n(C3CC3)c12",  433.0),
    ("H",       "CC(C)(C)Nc1ncnc2nc(-c3ccc(OF)cc3)n(C3CC3)c12",  437.0),  # same as seed
    ("I",       "CN(c1ncnc2nc(-c3ccc(OF)cc3)n(C3CC3)c12)C(C)(C)C", 541.0),
]

print("=== Chemistry Validity Re-check of All 9 Candidates ===\n")
print(f"{'ID':<6} {'Affinity':>10}  {'Status':<12}  {'Reason'}")
print("-" * 80)

valid_candidates = []
invalid_candidates = []

for label, smiles, affinity in CANDIDATES:
    is_valid, reason = check_smiles_validity(smiles)
    status = "✓ VALID" if is_valid else "✗ INVALID"
    reason_str = reason if reason else ""
    print(f"{label:<6} {affinity:>8.1f} nM  {status:<12}  {reason_str}")
    if is_valid:
        valid_candidates.append((label, smiles, affinity))
    else:
        invalid_candidates.append((label, smiles, affinity, reason))

print(f"\n{'='*80}")
print(f"Valid:   {len(valid_candidates)}/{len(CANDIDATES)}")
print(f"Invalid: {len(invalid_candidates)}/{len(CANDIDATES)}")

if invalid_candidates:
    print("\n⚠ INVALID CANDIDATES (must not be advanced to patent or CRO):")
    for label, smiles, affinity, reason in invalid_candidates:
        print(f"  {label}: {reason}")
        print(f"     SMILES: {smiles}")

if valid_candidates:
    print("\n✓ VALID CANDIDATES (pass chemistry filter):")
    for label, smiles, affinity in valid_candidates:
        mol = Chem.MolFromSmiles(smiles)
        mw = Descriptors.MolWt(mol) if mol else 0
        print(f"  {label}: {affinity:.1f} nM predicted, MW={mw:.1f}")
        print(f"     SMILES: {smiles}")
    
    # Check affinity threshold (≤50 nM for HIV protease)
    strong = [(l, s, a) for l, s, a in valid_candidates if a <= 50.0]
    if strong:
        print(f"\n★ Candidates meeting ≤50 nM threshold: {len(strong)}")
        for l, s, a in strong:
            print(f"  {l}: {a:.1f} nM")
    else:
        best = min(valid_candidates, key=lambda x: x[2])
        print(f"\n⚠ No valid candidates meet the ≤50 nM threshold.")
        print(f"  Best valid candidate: {best[0]} at {best[2]:.1f} nM")
        print(f"  The loop must continue running to find stronger binders.")
        print(f"  Current best is {best[2]/1.0:.0f}x weaker than darunavir (1 nM).")
