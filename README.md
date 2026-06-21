# ASI-Evolve Discovery Engine

An end-to-end AI-driven molecular drug discovery platform with a three-agent evolutionary optimization loop, automated molecular docking validation, and ADMET profiling.

## Architecture

```
                    +---------------------+
                    |   ChEMBL Database   |
                    +----------+----------+
                               |
                    +----------v----------+
                    |   Backend Core      |
                    |   (ChEMBL Client)   |
                    +----------+----------+
                               |
                    +----------v----------+
                    |  Morgan Fingerprints |
                    |  + RandomForest     |
                    |    Affinity Model    |
                    +----------+----------+
                               |
              +----------------v------------------+
              |         ASI-Evolve Loop          |
              |  +------------+  +------------+  |
              |  | Researcher |->|  Engineer  |  |
              |  |   Agent    |  |   Agent    |  |
              |  +------+-----+  +------+-----+  |
              |         |               |        |
              |         +------v--------+        |
              |                |                 |
              |         +------v-----+           |
              |         |  Analyzer  |           |
              |         |   Agent    |           |
              |         +------+-----+           |
              |                |                 |
              |         Cognition Store          |
              +----------------+-----------------+
                               |
              +----------------v------------------+
              |      Validation Pipeline          |
              |  +------------+ +------------+   |
              |  | AutoDock   | | SwissADME  |   |
              |  |   Vina     | |  Profile   |   |
              |  +------------+ +------------+   |
              +----------------+-----------------+
                               |
              +----------------v------------------+
              |     Discoveries Database         |
              +----------------+-----------------+
                               |
         +---------------------+---------------------+
         |                                           |
+--------v---------+                     +-----------v--------+
|  Evidence PDFs   |                     |   React Frontend   |
|  (Product)       |                     |   (Storefront)     |
+------------------+                     +--------------------+
```

## System Layers

### Layer 1: Backend Core Engine
- **ChEMBL Client**: Fetches binding affinity data (IC50) from the ChEMBL REST API for a defined target protein
- **Fingerprinting**: Converts SMILES molecular representations into 2048-bit Morgan fingerprints (ECFP4) using RDKit
- **ML Model**: Trains a RandomForest regressor to predict binding affinity (nM) from molecular fingerprints
- **Predictor**: Provides `predict(smiles)` API for end-to-end affinity prediction

### Layer 2: ASI-Evolve Agent Loop
Three-agent evolutionary optimization loop running 20 cycles/day:

| Agent | Role | Function |
|-------|------|----------|
| **Researcher** | Strategy | Reads Cognition Store, identifies patterns in high-scoring modifications, proposes next molecular change |
| **Engineer** | Execution | Implements modifications as numerical fingerprint operations (bit flips, guided mutations, crossovers) |
| **Analyzer** | Evaluation | Scores candidates, compares to current best, distills lessons into the Cognition Store |

**Cognition Store**: The system's accumulated knowledge — a structured log of every cycle, statistical patterns per fingerprint bit, and distilled lessons. This is the trade secret that enables progressive improvement.

### Layer 3: Validation + Presentation
- **AutoDock Vina**: Molecular docking calculations against target protein structures from PDB
- **SwissADME**: Full ADMET profiling (drug-likeness, toxicity, absorption, distribution)
- **Evidence PDFs**: Structured documents with complete evidence chains for each validated candidate
- **Public Website**: React frontend showing the discovery feed with real-time loop status

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (for full stack)
- Optional: AutoDock Vina (for molecular docking)

### Option 1: Docker Compose (Recommended)
```bash
# Clone and enter the project directory
cd asi-evolve

# Build and start all services
make build
make up

# Access the system
# API:     http://localhost:8000
# Docs:    http://localhost:8000/docs
# Web UI:  http://localhost:5173
```

### Option 2: Local Development
```bash
# Install dependencies
make setup

# Train the affinity prediction model
make train

# Start the API server
uvicorn backend.main:app --reload --port 8000

# In another terminal, start the frontend
cd frontend && npm install && npm run dev

# To run the optimization loop
python -m backend.agents.loop_scheduler
```

## Configuration

All settings are managed through `backend/config.py` and can be overridden via environment variables with the `MDE_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `MDE_TARGET_CHEMBL_ID` | CHEMBL203 | ChEMBL target ID (default: EGFR) |
| `MDE_ACTIVITY_TYPE` | IC50 | Bioactivity measurement type |
| `MDE_ACTIVITY_LIMIT` | 5000 | Max training records from ChEMBL |
| `MDE_FINGERPRINT_NBITS` | 2048 | Morgan fingerprint bit length |
| `MDE_N_ESTIMATORS` | 200 | Random Forest tree count |
| `MDE_MODEL_PATH` | data/model.pkl | Trained model save path |
| `MDE_DATA_DIR` | data/ | Local data directory |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/discoveries` | List all validated discoveries |
| GET | `/api/discoveries/{id}` | Single discovery detail |
| GET | `/api/discoveries/{id}/pdf` | Download evidence PDF |
| GET | `/api/loop/status` | Current loop status |
| POST | `/api/loop/start` | Start the optimization loop |
| POST | `/api/loop/stop` | Stop the loop |
| POST | `/api/loop/step` | Run one cycle manually |
| GET | `/api/loop/cognition` | Full cognition store data |
| GET | `/api/candidates` | List candidate molecules |
| GET | `/api/candidates/top` | Top N candidates by affinity |
| POST | `/api/candidates/evaluate` | Evaluate a SMILES string |
| GET | `/health` | Health check |

## Project Structure

```
asi-evolve/
├── backend/                    # Python backend
│   ├── main.py                 # FastAPI application entry
│   ├── config.py               # Configuration management
│   ├── core/                   # Layer 1: ML engine
│   │   ├── chembl_client.py    # ChEMBL API client
│   │   ├── fingerprint.py      # Morgan fingerprinting
│   │   ├── model_trainer.py    # RandomForest training
│   │   └── predictor.py        # Affinity prediction
│   ├── agents/                 # Layer 2: Agent loop
│   │   ├── cognition_store.py  # Knowledge repository
│   │   ├── researcher.py       # Strategy agent
│   │   ├── engineer.py         # Execution agent
│   │   ├── analyzer.py         # Evaluation agent
│   │   └── loop_scheduler.py  # Loop orchestrator
│   ├── validation/             # Layer 3: Quality gates
│   │   ├── vina_docker.py     # AutoDock Vina wrapper
│   │   ├── swissadme_client.py # ADMET profiling
│   │   └── validator.py        # Validation orchestrator
│   ├── database/               # Persistence layer
│   │   ├── models.py           # SQLAlchemy ORM
│   │   ├── discovery_db.py     # CRUD operations
│   │   └── session.py          # Database sessions
│   ├── evidence/               # PDF evidence generation
│   │   └── evidence_builder.py # Document builder
│   └── api/                    # REST API routers
│       ├── discovery.py
│       ├── loop_status.py
│       ├── candidates.py
│       └── evidence.py
├── frontend/                   # React web application
│   ├── src/
│   │   ├── components/         # UI components
│   │   ├── pages/              # Route pages
│   │   ├── hooks/              # Custom React hooks
│   │   └── types/              # TypeScript types
│   └── package.json
├── scripts/
│   └── train_model.py          # One-shot model training
├── tests/
│   └── test_pipeline.py        # Core engine tests
├── docker-compose.yml          # Full stack deployment
├── Dockerfile.backend          # API container
├── Dockerfile.frontend         # Web UI container
├── Makefile                    # Common commands
└── requirements.txt            # Python dependencies
```

## How It Works

### The Factory (Agent Loop)
1. **Initialization**: The system fetches binding affinity data from ChEMBL for the target protein, converts molecules to fingerprints, and trains the prediction model.

2. **Cycle Execution** (every ~72 minutes, 20x/day):
   - **Researcher** analyzes the Cognition Store's accumulated patterns and proposes a molecular modification strategy
   - **Engineer** applies the modification as a numerical operation on the fingerprint
   - **Analyzer** scores the new candidate, compares it to the current best, and distills a lesson
   - The Cognition Store grows with each cycle; candidates progressively improve

3. **Threshold Trigger**: Candidates scoring better than the best training compound advance to validation.

### The Quality Gate (Validation)
- **AutoDock Vina**: Confirms binding prediction through molecular docking against the target's 3D structure
- **SwissADME**: Verifies drug-likeness and checks for toxicity flags
- Both checks must pass for entry into the discoveries database.

### The Storefront (Website)
Each discovery entry displays:
- Candidate identifier (e.g., `ASI-20240621-0001`)
- Target protein and predicted binding affinity
- Docking score with pass/fail badge
- ADMET summary (drug-likeness, absorption, toxicity)
- Confidence score (composite 0-1)
- Link to the full evidence PDF

### The Product (Evidence PDF)
A structured document containing:
- Complete provenance chain from training data to final candidate
- Full fingerprint representation and modification history
- Docking results with binding mode analysis
- Comprehensive ADMET profile with pass/fail indicators
- Novelty statement comparing to known compounds
- Version number, date, and licensor attribution

## Testing

```bash
# Core engine tests
make test-core

# All tests
make test

# With coverage
pytest tests/ --cov=backend --cov-report=html
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI, Uvicorn |
| Cheminformatics | RDKit |
| Machine Learning | scikit-learn (RandomForest) |
| Database | SQLite (via SQLAlchemy 2.0) |
| Molecular Docking | AutoDock Vina |
| ADMET | SwissADME (web + RDKit fallback) |
| PDF Generation | WeasyPrint / fpdf2 |
| Frontend | React 18, TypeScript, Tailwind CSS, Vite |
| Real-time | WebSocket |
| Containerization | Docker, Docker Compose |

## License

Copyright (c) 2024 ASI-Evolve Discovery Systems. All rights reserved.

## Acknowledgments

- [ChEMBL](https://www.ebi.ac.uk/chembl/) — Bioactivity database
- [RDKit](https://www.rdkit.org/) — Cheminformatics toolkit
- [AutoDock Vina](https://vina.scripps.edu/) — Molecular docking
- [SwissADME](http://www.swissadme.ch/) — ADMET predictions
- [Protein Data Bank](https://www.rcsb.org/) — Target structures
