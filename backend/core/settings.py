"""Configuration settings for the ASI-Evolve agent loop."""

import os

# Fingerprint settings
FP_SIZE = int(os.getenv("ASI_FP_SIZE", "2048"))
FP_RADIUS = int(os.getenv("ASI_FP_RADIUS", "2"))

# Affinity threshold for triggering validation (nM)
# Candidates predicted below this value trigger experimental validation
AFFINITY_THRESHOLD_NM = float(os.getenv("ASI_AFFINITY_THRESHOLD_NM", "10.0"))

# Cycle timing
CYCLE_INTERVAL_SECONDS = float(
    os.getenv("ASI_CYCLE_INTERVAL_SECONDS", str(72 * 60))  # 72 min = 20 cycles/day
)

# Cognition store
COGNITION_STORE_PATH = os.getenv(
    "ASI_COGNITION_STORE_PATH", "./data/cognition_store.json"
)

# Logging
LOG_LEVEL = os.getenv("ASI_LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv(
    "ASI_LOG_FORMAT",
    "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
)

# Target (default: EGFR for testing)
DEFAULT_TARGET_CHEMBL_ID = os.getenv("ASI_TARGET_CHEMBL_ID", "CHEMBL203")
DEFAULT_TARGET_NAME = os.getenv("ASI_TARGET_NAME", "EGFR")
DEFAULT_PARENT_SMILES = os.getenv(
    "ASI_PARENT_SMILES",
    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",  # Gefitinib-like
)

# Cycle limits
MAX_CYCLES = int(os.getenv("ASI_MAX_CYCLES", "0"))  # 0 = unlimited
