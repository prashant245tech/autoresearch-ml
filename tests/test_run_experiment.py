import json
from pathlib import Path

import run_experiment
from tests.support import WorkspaceTestCase


VALID_TRAIN_SOURCE = """
import pandas as pd
from sklearn.linear_model import LinearRegression

EXPERIMENT_DESCRIPTION = "change_type=family_probe | family=LinearRegression | change=baseline"

def engineer_features(df, meta):
    features = pd.DataFrame(index=df.index)
    for col in meta.get("model_features", []):
        if col in df.columns:
            features[col] = df[col]
    return features

def build_model(meta):
    return LinearRegression()
"""


INVALID_PREDICT_TRAIN_SOURCE = """
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

EXPERIMENT_DESCRIPTION = "change_type=invalid | family=LinearRegression | change=bad_predict_shape"

def engineer_features(df, meta):
    features = pd.DataFrame(index=df.index)
    for col in meta.get("model_features", []):
        if col in df.columns:
            features[col] = df[col]
    return features

def build_model(meta):
    return LinearRegression()

def predict_model(model, X):
    return np.zeros(len(X) + 1)
"""


class RunExperimentTests(WorkspaceTestCase):
    def test_ensure_prepared_data_baseline_creates_and_enforces_session_hashes(self):
        baseline_train_df, _test_df, _meta = self.create_prepared_data()

        baseline = run_experiment.ensure_prepared_data_baseline()

        self.assertTrue(self.abs_path("experiments/session_baseline.json").exists())
        self.assertIn("prepared_data_sha", baseline)

        mutated_train = baseline_train_df.copy()
        mutated_train.loc[0, "price_msf"] += 1.0
        self.write_parquet("data/train.parquet", mutated_train)

        with self.assertRaises(run_experiment.PreparedDataMismatchError):
            run_experiment.ensure_prepared_data_baseline()

    def test_run_once_returns_validation_summary_without_saving_artifacts(self):
        self.create_prepared_data()
        self.write_text("train.py", VALID_TRAIN_SOURCE)

        summary, exit_code = run_experiment.run_once()

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["status"], run_experiment.STATUS_OK)
        self.assertEqual(summary["model_class"], "LinearRegression")
        self.assertIn("train_mape", summary)
        self.assertIn("val_mape", summary)
        self.assertIn("model_params", summary)
        self.assertTrue(self.abs_path("experiments/session_baseline.json").exists())
        self.assertFalse(self.abs_path("models/accepted").exists())

    def test_run_once_reports_invalid_candidate_for_bad_prediction_shape(self):
        self.create_prepared_data()
        self.write_text("train.py", INVALID_PREDICT_TRAIN_SOURCE)

        summary, exit_code = run_experiment.run_once()

        self.assertEqual(exit_code, 1)
        self.assertEqual(summary["status"], run_experiment.STATUS_INVALID_CANDIDATE)
        self.assertIn("returned", summary["error"])
        self.assertIn("predictions", summary["error"])

    def test_accept_once_rejects_mismatched_train_sha(self):
        self.create_prepared_data()
        self.write_text("train.py", VALID_TRAIN_SOURCE)

        summary, exit_code = run_experiment.accept_once("deadbeef")

        self.assertEqual(exit_code, 1)
        self.assertEqual(summary["status"], run_experiment.STATUS_HASH_MISMATCH)
        self.assertIn("does not match expected", summary["error"])

    def test_accept_once_saves_artifact_bundle(self):
        self.create_prepared_data()
        self.write_text("train.py", VALID_TRAIN_SOURCE)

        run_summary, run_exit_code = run_experiment.run_once()
        self.assertEqual(run_exit_code, 0)

        accept_summary, accept_exit_code = run_experiment.accept_once(run_summary["train_sha"])

        self.assertEqual(accept_exit_code, 0)
        self.assertEqual(accept_summary["status"], run_experiment.STATUS_OK)
        self.assertIn("test_mape", accept_summary)
        output_dir = Path(accept_summary["output_dir"])
        self.assertTrue((output_dir / "model.pkl").exists())
        self.assertTrue((output_dir / "feature_columns.json").exists())
        self.assertTrue((output_dir / "train.py").exists())
        self.assertTrue((output_dir / "manifest.json").exists())

        manifest = json.loads((output_dir / "manifest.json").read_text())
        self.assertEqual(manifest["train_sha"], run_summary["train_sha"])
        self.assertEqual(manifest["status"], run_experiment.STATUS_OK)
