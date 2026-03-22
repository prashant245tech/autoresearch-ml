import json
from pathlib import Path

import run_experiment
import search_memory
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
    def load_search_events(self):
        path = self.abs_path(search_memory.SEARCH_MEMORY_PATH)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def load_search_summary(self):
        path = self.abs_path(search_memory.SEARCH_SUMMARY_PATH)
        self.assertTrue(path.exists())
        return json.loads(path.read_text())

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
        events = self.load_search_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], search_memory.EVENT_TYPE_RUN)
        self.assertEqual(events[0]["status"], run_experiment.STATUS_OK)
        self.assertTrue(events[0]["candidate_signature"])
        search_summary = self.load_search_summary()
        self.assertEqual(search_summary["counts"]["total_runs"], 1)
        self.assertEqual(search_summary["counts"]["successful_runs"], 1)

    def test_run_once_reports_invalid_candidate_for_bad_prediction_shape(self):
        self.create_prepared_data()
        self.write_text("train.py", INVALID_PREDICT_TRAIN_SOURCE)

        summary, exit_code = run_experiment.run_once()

        self.assertEqual(exit_code, 1)
        self.assertEqual(summary["status"], run_experiment.STATUS_INVALID_CANDIDATE)
        self.assertIn("returned", summary["error"])
        self.assertIn("predictions", summary["error"])
        events = self.load_search_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], search_memory.EVENT_TYPE_RUN)
        self.assertEqual(events[0]["status"], run_experiment.STATUS_INVALID_CANDIDATE)
        self.assertIn("error", events[0])
        self.assertIsNone(events[0]["candidate_signature"])

    def test_accept_once_rejects_mismatched_train_sha(self):
        self.create_prepared_data()
        self.write_text("train.py", VALID_TRAIN_SOURCE)

        summary, exit_code = run_experiment.accept_once("deadbeef")

        self.assertEqual(exit_code, 1)
        self.assertEqual(summary["status"], run_experiment.STATUS_HASH_MISMATCH)
        self.assertIn("does not match expected", summary["error"])
        self.assertFalse(self.abs_path(search_memory.SEARCH_MEMORY_PATH).exists())

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
        events = self.load_search_events()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["event_type"], search_memory.EVENT_TYPE_RUN)
        self.assertEqual(events[1]["event_type"], search_memory.EVENT_TYPE_ACCEPT)
        self.assertEqual(events[1]["status"], run_experiment.STATUS_OK)
        self.assertEqual(events[1]["output_dir"], accept_summary["output_dir"])
        self.assertIn("test_mape", events[1])
        search_summary = self.load_search_summary()
        self.assertEqual(search_summary["counts"]["accepts"], 1)
        self.assertEqual(search_summary["counts"]["failed_accepts"], 0)

    def test_accept_workflow_failure_does_not_record_search_memory(self):
        self.create_prepared_data()
        self.write_text("train.py", VALID_TRAIN_SOURCE)

        run_summary, run_exit_code = run_experiment.run_once()
        self.assertEqual(run_exit_code, 0)

        existing_dir = self.abs_path(f"models/accepted/{run_summary['train_sha'][:12]}")
        existing_dir.mkdir(parents=True, exist_ok=True)

        accept_summary, accept_exit_code = run_experiment.accept_once(run_summary["train_sha"])

        self.assertEqual(accept_exit_code, 1)
        self.assertEqual(accept_summary["status"], run_experiment.STATUS_TRAIN_FAILED)
        self.assertIn("already exists", accept_summary["error"])
        events = self.load_search_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], search_memory.EVENT_TYPE_RUN)
        search_summary = self.load_search_summary()
        self.assertEqual(search_summary["counts"]["accepts"], 0)
        self.assertEqual(search_summary["counts"]["failed_accepts"], 0)

    def test_run_once_prepared_data_mismatch_does_not_record_search_memory(self):
        baseline_train_df, _test_df, _meta = self.create_prepared_data()
        self.write_text("train.py", VALID_TRAIN_SOURCE)

        baseline_run, baseline_exit = run_experiment.run_once()
        self.assertEqual(baseline_exit, 0)
        self.assertEqual(len(self.load_search_events()), 1)

        mutated_train = baseline_train_df.copy()
        mutated_train.loc[0, "price_msf"] += 10.0
        self.write_parquet("data/train.parquet", mutated_train)

        summary, exit_code = run_experiment.run_once()

        self.assertEqual(exit_code, 1)
        self.assertEqual(summary["status"], run_experiment.STATUS_PREPARED_DATA_MISMATCH)
        self.assertEqual(len(self.load_search_events()), 1)

    def test_memory_summary_returns_empty_valid_summary_when_no_history_exists(self):
        self.create_prepared_data()

        summary, exit_code = run_experiment.memory_summary_once()

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["counts"]["total_events"], 0)
        self.assertEqual(summary["counts"]["total_runs"], 0)
        self.assertEqual(summary["counts"]["accepts"], 0)
        self.assertEqual(summary["recent_events"], [])
        self.assertTrue(self.abs_path(search_memory.SEARCH_SUMMARY_PATH).exists())
        self.assertFalse(self.abs_path(search_memory.SEARCH_MEMORY_PATH).exists())

    def test_repeated_identical_runs_are_collapsed_in_summary(self):
        self.create_prepared_data()
        self.write_text("train.py", VALID_TRAIN_SOURCE)

        first_summary, first_exit = run_experiment.run_once()
        second_summary, second_exit = run_experiment.run_once()

        self.assertEqual(first_exit, 0)
        self.assertEqual(second_exit, 0)
        self.assertEqual(first_summary["train_sha"], second_summary["train_sha"])

        events = self.load_search_events()
        self.assertEqual(len(events), 2)
        search_summary = self.load_search_summary()
        self.assertEqual(search_summary["counts"]["total_runs"], 2)
        self.assertEqual(len(search_summary["top_unique_candidates"]), 1)
        candidate_signature = events[0]["candidate_signature"]
        self.assertEqual(
            search_summary["duplicate_candidate_counts"][candidate_signature],
            2,
        )
        self.assertEqual(search_summary["repeated_exact_runs"][0]["attempt_count"], 2)
        self.assertEqual(
            search_summary["repeated_exact_runs"][0]["candidate_signature"],
            candidate_signature,
        )

    def test_search_summary_is_scoped_to_current_prepared_data_sha(self):
        baseline_train_df, _test_df, _meta = self.create_prepared_data()
        self.write_text("train.py", VALID_TRAIN_SOURCE)

        first_summary, first_exit = run_experiment.run_once()
        self.assertEqual(first_exit, 0)
        first_scope = first_summary["prepared_data_sha"]

        self.abs_path("experiments/session_baseline.json").unlink()
        mutated_train = baseline_train_df.copy()
        mutated_train.loc[:, "price_msf"] = mutated_train["price_msf"] + 25.0
        self.write_parquet("data/train.parquet", mutated_train)

        second_summary, second_exit = run_experiment.run_once()
        self.assertEqual(second_exit, 0)
        second_scope = second_summary["prepared_data_sha"]
        self.assertNotEqual(first_scope, second_scope)

        summary, exit_code = run_experiment.memory_summary_once()
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["prepared_data_sha"], second_scope)
        self.assertEqual(summary["counts"]["total_runs"], 1)
        self.assertEqual(summary["best_run"]["train_sha"], second_summary["train_sha"])
