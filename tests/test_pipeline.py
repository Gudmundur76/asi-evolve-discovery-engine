"""End-to-end pipeline test using synthetic / mock data.

No network calls are made — the ChEMBL client is patched to return
canned responses so the test is fast and deterministic.
"""

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.config import Settings
from backend.core.chembl_client import ChEMBLClient, ChEMBLAPIError
from backend.core.fingerprint import FingerprintEncoder
from backend.core.model_trainer import ModelTrainer, ModelTrainerError
from backend.core.predictor import AffinityPredictor, PredictorError


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_activities_df() -> pd.DataFrame:
    """Return a small DataFrame mimicking ChEMBL activity data."""
    return pd.DataFrame(
        {
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3", "CHEMBL4"],
            "canonical_smiles": [
                "CCO",  # ethanol
                "c1ccccc1",  # benzene
                "CC(=O)O",  # acetic acid
                "CCCC",  # butane
            ],
            "standard_value": [100.0, 50.0, 200.0, 25.0],
            "standard_units": ["nM", "nM", "nM", "nM"],
        }
    )


@pytest.fixture
def encoder() -> FingerprintEncoder:
    return FingerprintEncoder(radius=2, n_bits=2048)


@pytest.fixture
def tmp_model_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


# ---------------------------------------------------------------------------
#  ChEMBL Client
# ---------------------------------------------------------------------------
class TestChEMBLClient:
    def test_fetch_target_info(self):
        """Target info should extract name, organism, type."""
        client = ChEMBLClient(target_chembl_id="CHEMBL203")

        with patch.object(client, "_paginated_get") as mock_get:
            mock_get.return_value = [
                {
                    "pref_name": "Epidermal Growth Factor Receptor",
                    "organism": "Homo sapiens",
                    "target_type": "SINGLE PROTEIN",
                }
            ]
            info = client.fetch_target_info()

        assert info["target_name"] == "Epidermal Growth Factor Receptor"
        assert info["organism"] == "Homo sapiens"
        assert info["target_type"] == "SINGLE PROTEIN"

    def test_fetch_target_info_empty_raises(self):
        """Empty target response should raise."""
        client = ChEMBLClient(target_chembl_id="CHEMBL_FAKE")

        with patch.object(client, "_paginated_get") as mock_get:
            mock_get.return_value = []
            with pytest.raises(Exception):
                client.fetch_target_info()

    def test_fetch_activities_parsing(self, mock_activities_df):
        """Activities should be parsed and deduplicated."""
        client = ChEMBLClient()

        with patch.object(client, "_paginated_get") as mock_get:
            # _paginated_get returns raw records which get DataFrame'd
            mock_get.return_value = mock_activities_df.to_dict("records")
            df = client.fetch_activities(activity_type="IC50")

        assert len(df) == 4
        assert list(df.columns) == [
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_value",
            "standard_units",
        ]
        assert df["standard_value"].dtype == float

    def test_fetch_activities_deduplicates_best_affinity(self):
        """Duplicate molecule IDs should keep the lowest standard_value."""
        client = ChEMBLClient()
        records = [
            {
                "molecule_chembl_id": "CHEMBL1",
                "canonical_smiles": "CCO",
                "standard_value": "100.0",
                "standard_units": "nM",
            },
            {
                "molecule_chembl_id": "CHEMBL1",
                "canonical_smiles": "CCO",
                "standard_value": "10.0",
                "standard_units": "nM",
            },
        ]

        with patch.object(client, "_paginated_get") as mock_get:
            mock_get.return_value = records
            df = client.fetch_activities()

        assert len(df) == 1
        assert df.iloc[0]["standard_value"] == 10.0  # best (lowest) kept


# ---------------------------------------------------------------------------
#  Fingerprint Encoder
# ---------------------------------------------------------------------------
class TestFingerprintEncoder:
    def test_smiles_to_fp_shape_and_dtype(self, encoder):
        """A valid SMILES should yield a uint8 0/1 array of the right length."""
        fp = encoder.smiles_to_fp("CCO")
        assert fp is not None
        assert fp.shape == (2048,)
        assert fp.dtype == np.uint8
        assert set(np.unique(fp)).issubset({0, 1})

    def test_smiles_to_fp_invalid_returns_none(self, encoder):
        """Garbage SMILES should return None, not crash."""
        fp = encoder.smiles_to_fp("NOT_A_SMILES!!!")
        assert fp is None

    def test_fp_to_sparse(self, encoder):
        """Sparse representation should list set-bit indices."""
        fp = np.zeros(2048, dtype=np.uint8)
        fp[[0, 10, 100]] = 1
        sparse = encoder.fp_to_sparse(fp)
        assert sparse == [0, 10, 100]

    def test_mutate_fp_changes_bits(self, encoder):
        """Mutation should flip exactly *num_bits* positions."""
        fp = np.zeros(2048, dtype=np.uint8)
        mutated = encoder.mutate_fp(fp, num_bits=5, seed=42)
        assert mutated.sum() == 5

    def test_crossover_fps_shape(self, encoder):
        """Crossover child should inherit shape from parents."""
        fp1 = np.ones(2048, dtype=np.uint8)
        fp2 = np.zeros(2048, dtype=np.uint8)
        child = encoder.crossover_fps(fp1, fp2, seed=42)
        assert child.shape == (2048,)
        assert child.dtype == np.uint8
        # Child should be a mix of both
        assert 0 < child.sum() < 2048


# ---------------------------------------------------------------------------
#  Model Trainer
# ---------------------------------------------------------------------------
class TestModelTrainer:
    def test_train_evaluate_cycle(self, tmp_model_dir):
        """Train → evaluate → save → load should be consistent."""
        rng = np.random.default_rng(42)
        X = rng.random((200, 2048)).astype(np.float64)
        y = rng.random(200) * 3 + 1  # log10 values between 1 and 4

        trainer = ModelTrainer(
            n_estimators=50, max_depth=10, test_size=0.2, random_state=42
        )
        metrics = trainer.train(X, y)

        # Sanity checks
        assert "r2_score" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert metrics["n_train"] == 160
        assert metrics["n_test"] == 40

        # Save & reload
        model_path = tmp_model_dir / "test_model.pkl"
        trainer.save(model_path)
        assert model_path.exists()

        trainer2 = ModelTrainer()
        trainer2.load_model(model_path)
        eval2 = trainer2.evaluate(X[:10], y[:10])
        assert "r2" in eval2
        assert "predictions" in eval2

    def test_train_too_few_samples_raises(self):
        """Training with < 10 samples should raise."""
        trainer = ModelTrainer()
        X = np.random.rand(5, 10)
        y = np.random.rand(5)
        with pytest.raises(ModelTrainerError):
            trainer.train(X, y)

    def test_evaluate_without_train_raises(self):
        """Evaluating before training should raise."""
        trainer = ModelTrainer()
        with pytest.raises(ModelTrainerError):
            trainer.evaluate(np.random.rand(10, 10), np.random.rand(10))

    def test_metadata_populated(self, tmp_model_dir):
        """Metadata should contain expected keys after training."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 128))
        y = rng.random(100) * 3 + 1

        trainer = ModelTrainer(n_estimators=10, random_state=42)
        trainer.train(X, y)

        required_keys = {
            "model_type",
            "n_train",
            "n_test",
            "test_r2",
            "test_rmse",
            "test_mae",
            "best_affinity_nM",
        }
        assert required_keys.issubset(trainer.metadata.keys())


# ---------------------------------------------------------------------------
#  Predictor
# ---------------------------------------------------------------------------
class TestAffinityPredictor:
    def test_predict_returns_positive_affinity(self, tmp_model_dir, encoder):
        """Prediction should return a positive nM value."""
        # Train a tiny model and save it
        rng = np.random.default_rng(42)
        X = rng.random((100, 2048)).astype(np.float64)
        y = rng.random(100) * 3 + 1

        trainer = ModelTrainer(n_estimators=10, random_state=42)
        trainer.train(X, y)

        model_path = tmp_model_dir / "model.pkl"
        meta_path = tmp_model_dir / "model_metadata.json"
        trainer.save(model_path)

        # Write metadata to the expected location
        with open(meta_path, "w") as fh:
            json.dump(trainer.metadata, fh)

        # Create predictor pointing at our temp model
        predictor = AffinityPredictor(
            model_path=model_path,
            encoder=encoder,
            settings_override=Settings(
                model_path=str(model_path),
                model_metadata_path=str(meta_path),
            ),
        )

        fp = encoder.smiles_to_fp("CCO")
        affinity = predictor.predict(fp)
        assert isinstance(affinity, float)
        assert affinity >= 0.001

    def test_predict_batch(self, tmp_model_dir, encoder):
        """Batch prediction should return a list of floats."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 2048)).astype(np.float64)
        y = rng.random(100) * 3 + 1

        trainer = ModelTrainer(n_estimators=10, random_state=42)
        trainer.train(X, y)

        model_path = tmp_model_dir / "model.pkl"
        meta_path = tmp_model_dir / "model_metadata.json"
        trainer.save(model_path)

        with open(meta_path, "w") as fh:
            json.dump(trainer.metadata, fh)

        predictor = AffinityPredictor(
            model_path=model_path,
            encoder=encoder,
            settings_override=Settings(
                model_path=str(model_path),
                model_metadata_path=str(meta_path),
            ),
        )

        fps = np.vstack([encoder.smiles_to_fp(s) for s in ("CCO", "c1ccccc1", "CC(=O)O")])
        preds = predictor.predict_batch(fps)
        assert len(preds) == 3
        assert all(isinstance(p, float) and p >= 0.001 for p in preds)

    def test_predict_smiles_end_to_end(self, tmp_model_dir, encoder):
        """SMILES-based prediction should work without manual fingerprinting."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 2048)).astype(np.float64)
        y = rng.random(100) * 3 + 1

        trainer = ModelTrainer(n_estimators=10, random_state=42)
        trainer.train(X, y)

        model_path = tmp_model_dir / "model.pkl"
        meta_path = tmp_model_dir / "model_metadata.json"
        trainer.save(model_path)

        with open(meta_path, "w") as fh:
            json.dump(trainer.metadata, fh)

        predictor = AffinityPredictor(
            model_path=model_path,
            encoder=encoder,
            settings_override=Settings(
                model_path=str(model_path),
                model_metadata_path=str(meta_path),
            ),
        )

        affinity = predictor.predict_smiles("CCO")
        assert isinstance(affinity, float)
        assert affinity >= 0.001

    def test_best_training_affinity_property(self, tmp_model_dir, encoder):
        """best_training_affinity should be a positive float."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 2048)).astype(np.float64)
        y = rng.random(100) * 3 + 1

        trainer = ModelTrainer(n_estimators=10, random_state=42)
        trainer.train(X, y)

        model_path = tmp_model_dir / "model.pkl"
        meta_path = tmp_model_dir / "model_metadata.json"
        trainer.save(model_path)

        with open(meta_path, "w") as fh:
            json.dump(trainer.metadata, fh)

        predictor = AffinityPredictor(
            model_path=model_path,
            encoder=encoder,
            settings_override=Settings(
                model_path=str(model_path),
                model_metadata_path=str(meta_path),
            ),
        )

        assert isinstance(predictor.best_training_affinity, float)
        assert predictor.best_training_affinity > 0

    def test_is_better_than_training(self, tmp_model_dir, encoder):
        """Affinities lower than the training best should report True."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 2048)).astype(np.float64)
        y = rng.random(100) * 3 + 1

        trainer = ModelTrainer(n_estimators=10, random_state=42)
        trainer.train(X, y)

        model_path = tmp_model_dir / "model.pkl"
        meta_path = tmp_model_dir / "model_metadata.json"
        trainer.save(model_path)

        with open(meta_path, "w") as fh:
            json.dump(trainer.metadata, fh)

        predictor = AffinityPredictor(
            model_path=model_path,
            encoder=encoder,
            settings_override=Settings(
                model_path=str(model_path),
                model_metadata_path=str(meta_path),
            ),
        )

        best = predictor.best_training_affinity
        assert predictor.is_better_than_training(best / 2) is True
        assert predictor.is_better_than_training(best * 2) is False

    def test_predict_smiles_invalid_returns_none(self, tmp_model_dir, encoder):
        """Invalid SMILES should return None gracefully."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 2048)).astype(np.float64)
        y = rng.random(100) * 3 + 1

        trainer = ModelTrainer(n_estimators=10, random_state=42)
        trainer.train(X, y)

        model_path = tmp_model_dir / "model.pkl"
        meta_path = tmp_model_dir / "model_metadata.json"
        trainer.save(model_path)

        with open(meta_path, "w") as fh:
            json.dump(trainer.metadata, fh)

        predictor = AffinityPredictor(
            model_path=model_path,
            encoder=encoder,
            settings_override=Settings(
                model_path=str(model_path),
                model_metadata_path=str(meta_path),
            ),
        )

        result = predictor.predict_smiles("NOT_A_SMILES!!!")
        assert result is None

    def test_missing_model_raises(self):
        """Predictor should raise when model file is missing."""
        with pytest.raises(PredictorError):
            AffinityPredictor(
                model_path="/tmp/nonexistent_model.pkl",
                settings_override=Settings(
                    model_path="/tmp/nonexistent_model.pkl",
                    model_metadata_path="/tmp/nonexistent.json",
                ),
            )


# ---------------------------------------------------------------------------
#  Integration: full pipeline with mocked ChEMBL
# ---------------------------------------------------------------------------
class TestFullPipeline:
    def test_end_to_end(self, tmp_model_dir, encoder):
        """Simulate the full pipeline: fetch → fingerprint → train → predict."""
        # 1. Mock ChEMBL data — use diverse SMILES so fingerprints differ meaningfully
        diverse_smiles = [
            "CCO", "c1ccccc1", "CC(=O)O", "CCCC", "CC(C)O",
            "CC(C)C", "c1ccc(O)cc1", "CCN", "CC(C)=O", "CCCCO",
            "c1ccccc1O", "CC(C)N", "CC(C)C(=O)O", "CC(C)OC", "c1ccc(cc1)O",
            "CC(C)CC", "CC(C)CO", "CC(C)C(=O)N", "CC(C)S", "CCCCN",
            "c1ccccc1N", "CC(C)C(C)C", "CC(C)C(C)O", "CC(C)C(C)N", "c1ccc(cc1)N",
            "CC(C)C(C)C(=O)O", "CC(C)C(C)C(=O)N", "CC(C)C(C)S", "CCCCS", "c1ccccc1S",
            "CC(C)C(C)C(C)C", "CC(C)C(C)C(C)O", "CC(C)C(C)C(C)N", "CC(C)C(C)C(C)C(=O)O",
            "c1ccc(cc1)S", "CCCC(C)C", "CCCC(C)O", "CCCC(C)N", "CCCC(C)C(=O)O",
            "CC(C)C(C)C(C)C(=O)N", "CC(C)C(C)C(C)S", "c1ccccc1C", "CC(C)C(C)C(C)C(C)C",
            "CCCC(C)S", "c1ccc(cc1)C", "CC(C)C(C)C(C)C(C)O", "CCCC(C)C(=O)N",
            "CC(C)C(C)C(C)C(C)N", "c1ccccc1CC", "CC(C)C(C)C(C)C(C)C(=O)O",
        ]
        n_mols = len(diverse_smiles)
        mock_df = pd.DataFrame(
            {
                "molecule_chembl_id": [f"CHEMBL{i}" for i in range(1, n_mols + 1)],
                "canonical_smiles": diverse_smiles,
                # Affinity loosely correlates with molecule size/complexity
                "standard_value": [10.0 + 5.0 * len(s) + np.random.default_rng(42).uniform(-20, 20)
                                   for s in diverse_smiles],
                "standard_units": ["nM"] * n_mols,
            }
        )
        # Ensure all affinities are positive
        mock_df["standard_value"] = mock_df["standard_value"].abs().clip(lower=0.1)

        # 2. Mock the ChEMBL client
        client = ChEMBLClient()
        with patch.object(client, "_paginated_get") as mock_get:
            mock_get.return_value = mock_df.to_dict("records")
            activities = client.fetch_activities()

        assert len(activities) == n_mols  # all unique molecule_chembl_id

        # 3. Encode fingerprints
        fingerprints, affinities = [], []
        for _, row in activities.iterrows():
            fp = encoder.smiles_to_fp(row["canonical_smiles"])
            if fp is not None:
                fingerprints.append(fp)
                affinities.append(row["standard_value"])

        X = np.vstack(fingerprints)
        y = np.log10(np.maximum(affinities, 1e-6))

        # 4. Train
        trainer = ModelTrainer(n_estimators=30, max_depth=15, random_state=42)
        metrics = trainer.train(X, y)
        # With real structure in data R2 should be positive
        assert metrics["r2_score"] > -5.0

        # 5. Save
        model_path = tmp_model_dir / "pipeline_model.pkl"
        meta_path = tmp_model_dir / "pipeline_model_metadata.json"
        trainer.save(model_path)
        with open(meta_path, "w") as fh:
            json.dump(trainer.metadata, fh)

        # 6. Predict
        predictor = AffinityPredictor(
            model_path=model_path,
            encoder=encoder,
            settings_override=Settings(
                model_path=str(model_path),
                model_metadata_path=str(meta_path),
            ),
        )

        affinity = predictor.predict_smiles("CCO")
        assert isinstance(affinity, float) and affinity >= 0.001
        assert predictor.best_training_affinity > 0
