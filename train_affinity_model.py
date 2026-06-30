#!/usr/bin/env python3
"""
Retrain the AffinityPredictor on real ChEMBL HIV-1 Protease data.

Downloads IC50/Ki measurements from CHEMBL243 (Human immunodeficiency virus
type 1 protease), computes Morgan fingerprints, trains a Random Forest
regressor on pChEMBL values (= -log10(IC50_M)), and saves the model to
data/model.pkl alongside updated metadata.

Usage:
    python train_affinity_model.py

Output:
    data/model.pkl          — trained RandomForestRegressor (sklearn)
    data/model_metadata.json — training stats and feature info
    data/training_data.csv  — cleaned dataset for reproducibility
"""

import json
import logging
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_CHEMBL_ID = "CHEMBL243"
TARGET_NAME = "HIV-1 Protease"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_SIZE = 1000
RADIUS = 2
N_BITS = 2048
MIN_PCHEMBL = 3.0   # discard very weak binders (pChEMBL < 3 = IC50 > 1 mM)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# ── Step 1: Download ──────────────────────────────────────────────────────────
def download_chembl_data() -> pd.DataFrame:
    """Fetch all pChEMBL-valued IC50/Ki records for TARGET_CHEMBL_ID."""
    log.info("Downloading ChEMBL data for %s (%s)…", TARGET_NAME, TARGET_CHEMBL_ID)
    records = []
    offset = 0

    while True:
        url = (
            f"{CHEMBL_API}/activity.json"
            f"?target_chembl_id={TARGET_CHEMBL_ID}"
            f"&pchembl_value__isnull=false"
            f"&limit={PAGE_SIZE}&offset={offset}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        activities = data["activities"]
        records.extend(activities)
        log.info("  Downloaded %d / %d records", len(records), data["page_meta"]["total_count"])

        if not data["page_meta"]["next"]:
            break
        offset += PAGE_SIZE
        time.sleep(0.3)   # be polite to EBI

    df = pd.DataFrame(records)
    log.info("Raw download: %d records", len(df))
    return df


# ── Step 2: Clean ─────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter, deduplicate, and validate the activity data."""
    # Keep only rows with SMILES and pChEMBL value
    df = df.dropna(subset=["canonical_smiles", "pchembl_value"])
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=["pchembl_value"])

    # Filter weak binders
    df = df[df["pchembl_value"] >= MIN_PCHEMBL]

    # Validate SMILES
    valid_mask = df["canonical_smiles"].apply(
        lambda s: Chem.MolFromSmiles(str(s)) is not None
    )
    df = df[valid_mask]
    log.info("After SMILES validation: %d records", len(df))

    # Deduplicate by SMILES — keep median pChEMBL per molecule
    df = (
        df.groupby("canonical_smiles")["pchembl_value"]
        .median()
        .reset_index()
        .rename(columns={"pchembl_value": "pchembl_median"})
    )
    log.info("After deduplication: %d unique molecules", len(df))

    # Convert pChEMBL back to nM for the predictor
    # pChEMBL = -log10(value_M)  →  value_M = 10^(-pChEMBL)  →  value_nM = 1e9 * 10^(-pChEMBL)
    df["affinity_nm"] = 1e9 * np.power(10, -df["pchembl_median"])

    return df


# ── Step 3: Featurise ─────────────────────────────────────────────────────────
def smiles_to_fp(smiles: str) -> np.ndarray | None:
    """Compute Morgan fingerprint as a dense numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=RADIUS, nBits=N_BITS)
    return np.array(fp, dtype=np.float32)


def featurise(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Convert SMILES to fingerprint matrix X and affinity vector y."""
    log.info("Computing Morgan fingerprints (r=%d, bits=%d)…", RADIUS, N_BITS)
    fps = []
    y = []
    for _, row in df.iterrows():
        fp = smiles_to_fp(row["canonical_smiles"])
        if fp is not None:
            fps.append(fp)
            y.append(row["pchembl_median"])   # train on pChEMBL (log scale)

    X = np.vstack(fps)
    y = np.array(y, dtype=np.float32)
    log.info("Feature matrix: %s, target vector: %s", X.shape, y.shape)
    return X, y


# ── Step 4: Train ─────────────────────────────────────────────────────────────
def train_model(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    """Train and cross-validate a Random Forest regressor."""
    log.info("Splitting data (80/20)…")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log.info("Training RandomForestRegressor (n_estimators=200)…")
    model = RandomForestRegressor(
        n_estimators=200,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred_test))

    log.info("Test set  — R²: %.3f  RMSE: %.3f pChEMBL units", r2, rmse)

    # 5-fold CV
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
    log.info("5-fold CV — R²: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

    return model, {
        "n_molecules": len(y),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "test_r2": round(float(r2), 4),
        "test_rmse_pchembl": round(float(rmse), 4),
        "cv_r2_mean": round(float(cv_scores.mean()), 4),
        "cv_r2_std": round(float(cv_scores.std()), 4),
        "target_chembl_id": TARGET_CHEMBL_ID,
        "target_name": TARGET_NAME,
        "fingerprint_radius": RADIUS,
        "fingerprint_n_bits": N_BITS,
        "trained_on": "pChEMBL (−log10 IC50/Ki in M)",
        "prediction_unit": "pChEMBL",
        "note": "Predictor returns pChEMBL. Convert to nM: affinity_nm = 1e9 * 10^(−pchembl)",
    }


# ── Step 5: Save ──────────────────────────────────────────────────────────────
def save_artifacts(model, stats: dict, df: pd.DataFrame) -> None:
    """Persist model, metadata, and training data."""
    model_path = DATA_DIR / "model.pkl"
    meta_path = DATA_DIR / "model_metadata.json"
    data_path = DATA_DIR / "training_data.csv"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info("Model saved → %s", model_path)

    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Metadata saved → %s", meta_path)

    df.to_csv(data_path, index=False)
    log.info("Training data saved → %s  (%d rows)", data_path, len(df))


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    raw_df = download_chembl_data()
    clean_df = clean_data(raw_df)
    clean_df.to_csv(DATA_DIR / "training_data.csv", index=False)

    X, y = featurise(clean_df)
    model, stats = train_model(X, y)
    save_artifacts(model, stats, clean_df)

    log.info("=" * 60)
    log.info("Training complete.")
    log.info("  Molecules: %d", stats["n_molecules"])
    log.info("  Test R²:   %.3f", stats["test_r2"])
    log.info("  CV R²:     %.3f ± %.3f", stats["cv_r2_mean"], stats["cv_r2_std"])
    log.info("  RMSE:      %.3f pChEMBL units", stats["test_rmse_pchembl"])
    log.info("=" * 60)
    log.info("Darunavir pChEMBL reference: ~11.5 (Ki = 0.003 nM)")
    log.info("Model output is pChEMBL — higher = better binder.")


if __name__ == "__main__":
    main()
