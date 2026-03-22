import json
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import explain
import run_experiment
from tests.support import DEFAULT_TARGET_COLUMN, WorkspaceTestCase


TREE_TRAIN_SOURCE = """
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

EXPERIMENT_DESCRIPTION = "move_intent=explore_new_branch | change_type=family_probe | family=RandomForest | change=baseline"

def engineer_features(df, meta):
    features = pd.DataFrame(index=df.index)
    for col in meta.get("model_features", []):
        if col in df.columns:
            features[col] = df[col]
    return features

def build_model(meta):
    return RandomForestRegressor(n_estimators=25, max_depth=4, random_state=42, n_jobs=1)
"""


class FakeTreeExplainer:
    def __init__(self, model):
        self.expected_value = 0.5

    def shap_values(self, X):
        return np.ones((len(X), X.shape[1]), dtype=float)


class ExplainTests(WorkspaceTestCase):
    def create_accepted_artifact(self):
        self.create_prepared_data()
        self.write_text("train.py", TREE_TRAIN_SOURCE)

        run_summary, run_exit = run_experiment.run_once()
        self.assertEqual(run_exit, 0)

        accept_summary, accept_exit = run_experiment.accept_once(run_summary["train_sha"])
        self.assertEqual(accept_exit, 0)
        return Path(accept_summary["output_dir"])

    def test_explain_reports_missing_shap_dependency(self):
        artifact_dir = self.create_accepted_artifact()

        with mock.patch.dict(sys.modules, {"shap": None}):
            summary, exit_code = explain.build_explanation_summary(
                artifact_dir=str(artifact_dir),
                dataset="test",
                sample_size=5,
                top_k=5,
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(summary["status"], explain.STATUS_MISSING_DEPENDENCY)
        self.assertIn("pip install shap", summary["error"])

    def test_build_feature_frame_uses_target_from_meta(self):
        train_spec = run_experiment.load_train_spec_from_source(TREE_TRAIN_SOURCE)
        df = pd.DataFrame({"x": [1.0, 2.0], "target_value": [10.0, 20.0]})
        meta = {"model_features": ["x"], "target": "target_value"}

        X = explain.build_feature_frame(train_spec, df, meta, ["x"])

        self.assertEqual(list(X.columns), ["x"])
        self.assertEqual(X["x"].tolist(), [1.0, 2.0])

    def test_explain_writes_summary_with_fake_shap_module(self):
        artifact_dir = self.create_accepted_artifact()
        fake_shap = types.SimpleNamespace(TreeExplainer=FakeTreeExplainer)

        with mock.patch.dict(sys.modules, {"shap": fake_shap}):
            summary, exit_code = explain.build_explanation_summary(
                artifact_dir=str(artifact_dir),
                dataset="test",
                sample_size=5,
                top_k=5,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["status"], explain.STATUS_OK)
        self.assertEqual(summary["dataset"], "test")
        self.assertEqual(summary["dataset_path"], run_experiment.TEST_DATA_PATH)
        self.assertEqual(summary["target_column"], DEFAULT_TARGET_COLUMN)
        self.assertEqual(summary["sample_size_used"], 5)
        self.assertEqual(summary["shap_backend"], "TreeExplainer")
        self.assertEqual(summary["shap_expected_value"], 0.5)
        self.assertEqual(summary["top_mean_abs_shap"], {"x": 1.0})
        self.assertTrue(summary["prepared_data_match"])

        output_path = Path(summary["output_path"])
        self.assertTrue(output_path.exists())
        written = json.loads(output_path.read_text())
        self.assertEqual(written["top_mean_abs_shap"], {"x": 1.0})
