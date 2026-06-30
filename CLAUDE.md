# CLAUDE.md — asi-evolve-discovery-engine
*Last updated: 2026-06-30 (Session: Sprint 3 + Chemistry Filter Fix)*

## What This Repo Does

ASI-Evolve is a **small-molecule drug candidate generation engine**. It runs a continuous loop that:
1. Starts from a known drug seed (e.g. darunavir for HIV-1 protease)
2. Generates chemically valid analogues using RDKit-based mutation strategies
3. Scores each candidate using a ChEMBL-trained Random Forest affinity predictor
4. Tracks the best-so-far candidate and detects plateau convergence
5. Generates an evidence PDF (WeasyPrint) for each new best candidate
6. Exposes a FastAPI REST interface on port 8001

## Architecture

```
backend/
  agents/
    loop_scheduler.py      ← Main loop orchestrator (run_single_cycle, get_status)
    smiles_mutator.py      ← RDKit mutation engine (5 strategies + chemistry filter)
    cognition_store.py     ← In-memory state: best candidate, seen-set (Tanimoto dedup)
    affinity_predictor.py  ← ChEMBL RF model wrapper (predict_smiles → float nM)
    evidence_builder.py    ← PDF generator (WeasyPrint + fpdf2 fallback)
  api/
    loop_router.py         ← FastAPI router: /api/loop/step, /api/loop/status
  main.py                  ← FastAPI app entry point
data/
  chembl243_rf_model.pkl   ← Trained sklearn RandomForestRegressor (4,719 records)
  model_metadata.json      ← Training provenance: R²=0.678, RMSE=0.886 log₁₀ nM
```

## Current State (2026-06-30)

**Working:**
- RDKit mutation engine: 5 strategies (exploration, guided_mutation, bit_flip, crossover, scaffold_hop)
- Chemistry validity filter: 20 SMARTS patterns rejecting O-F, N-F, peroxides, azides, acyl halides, hypervalent atoms, etc.
- Tanimoto InChIKey dedup: `cognition_store.is_seen()` / `mark_seen()` — retries up to 5× on duplicate
- Plateau detector: `_cycles_since_improvement` counter, overrides strategy to `guided_mutation` after 5 stale cycles
- EvidenceBuilder wired: generates PDF when new best candidate found, path stored in `_best_evidence_pdf`
- AffinityPredictor: returns real float (e.g. 45.7 nM for darunavir analogue)
- API: `/api/loop/step` runs one cycle, `/api/loop/status` returns full state

**Known issues:**
- The seed SMILES was previously `CC(C)(C)Nc1ncnc2nc(-c3ccc(OF)cc3)n(C3CC3)c12` — this contains a chemically impossible O-F bond. **Fixed**: chemistry filter now rejects this. Use darunavir as seed.
- Affinity model is ChEMBL243 (HIV-1 protease). For PCSK9/ANGPTL3 targets, the model gives meaningless predictions — a target-specific model is needed.
- EvidenceBuilder PDF path is stored in memory only — not persisted to DB or returned via API yet.

**Commits this session:**
- `b8b5ecd` — chemistry validity filter
- `0ebb055` — EvidenceBuilder wired + chemistry filter

## How to Run

```bash
cd backend
uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload
```

Test one cycle:
```bash
curl -X POST http://localhost:8001/api/loop/step \
  -H "Content-Type: application/json" \
  -d '{"strategy":"exploration","seed_smiles":"CC1(C)OC2C(OC(=O)c3ccccc3)C(OC(=O)c3ccccc3)OC2C1OC(=O)c1ccccc1"}'
```

## Next Steps

1. **PCSK9 target**: find a valid PCSK9 small-molecule seed from ChEMBL (CHEMBL5619 or CHEMBL5312) and run the loop — the patent landscape is less crowded than HIV protease
2. **Persist evidence PDF path**: return `evidence_pdf_url` in the `/api/loop/step` response so `generic-signal-api` can attach it to deliveries
3. **Wet-lab validation**: the 45.7 nM darunavir analogue needs one binding assay (~$500 from a CRO) to become a licensable asset
4. **Deploy**: Fly.io (`fly launch` from this directory) — Python/uvicorn fits perfectly

## Integration Points

- Called by `generic-signal-api` autonomous loop via HTTP POST to `/api/loop/step`
- `generic-signal-api` reads `new_smiles`, `predicted_affinity_nm`, `is_best_so_far`, `mutation_desc`
- Evidence PDF is generated locally — needs S3 upload to be accessible to partners

## Critical Rules

- **Never use a seed SMILES containing O-F, N-F, or other impossible groups** — the chemistry filter will reject all descendants
- **Always validate seeds** with `smiles_mutator.is_chemically_valid(smiles)` before starting a loop
- **The affinity model is target-specific** — it was trained on HIV-1 protease data only. Predictions for other targets are not meaningful.
