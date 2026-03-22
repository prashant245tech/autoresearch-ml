import json
from pathlib import Path

import run_experiment
import search_memory
from tests.support import WorkspaceTestCase


VALID_TRAIN_SOURCE = """
import pandas as pd
from sklearn.linear_model import LinearRegression

EXPERIMENT_DESCRIPTION = "move_intent=explore_new_branch | change_type=family_probe | family=LinearRegression | change=baseline"

def engineer_features(df, meta):
    features = pd.DataFrame(index=df.index)
    for col in meta.get("model_features", []):
        if col in df.columns:
            features[col] = df[col]
    return features

def build_model(meta):
    return LinearRegression()
"""


TREE_TRAIN_SOURCE = """
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

EXPERIMENT_DESCRIPTION = "move_intent=explore_new_branch | change_type=family_probe | family=RandomForest | change=tree_baseline"

def engineer_features(df, meta):
    features = pd.DataFrame(index=df.index)
    for col in meta.get("model_features", []):
        if col in df.columns:
            features[col] = df[col]
    return features

def build_model(meta):
    return RandomForestRegressor(n_estimators=25, max_depth=4, random_state=42, n_jobs=1)
"""


INVALID_PREDICT_TRAIN_SOURCE = """
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

EXPERIMENT_DESCRIPTION = "move_intent=exploit_current_winner | change_type=invalid | family=LinearRegression | change=bad_predict_shape"

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
    def write_baseline_train(self, content: str = VALID_TRAIN_SOURCE):
        return self.write_text("custom-baselines/train.test.py", content)

    def test_init_train_once_creates_local_train_from_baseline(self):
        baseline_path = self.write_baseline_train()

        summary, exit_code = run_experiment.init_train_once(
            baseline_path=str(baseline_path)
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["status"], run_experiment.STATUS_OK)
        self.assertEqual(summary["baseline_path"], "custom-baselines/train.test.py")
        self.assertTrue(self.abs_path("train.py").exists())
        self.assertEqual(self.abs_path("train.py").read_text(), VALID_TRAIN_SOURCE)

    def test_init_train_once_requires_force_to_replace_existing_train(self):
        baseline_path = self.write_baseline_train(VALID_TRAIN_SOURCE)
        self.write_text("train.py", TREE_TRAIN_SOURCE)

        summary, exit_code = run_experiment.init_train_once(
            baseline_path=str(baseline_path)
        )

        self.assertEqual(exit_code, 1)
        self.assertEqual(summary["status"], run_experiment.STATUS_INIT_FAILED)
        self.assertIn("already exists", summary["error"])
        self.assertEqual(self.abs_path("train.py").read_text(), TREE_TRAIN_SOURCE)

    def test_parse_experiment_description_extracts_structured_fields(self):
        parsed = search_memory.parse_experiment_description(
            "move_intent=explore_new_branch | change_type=family_probe | family=LinearRegression | change=baseline | hypothesis=test branch"
        )

        self.assertEqual(parsed["move_intent"], "explore_new_branch")
        self.assertEqual(parsed["change_type"], "family_probe")
        self.assertEqual(parsed["declared_family"], "LinearRegression")
        self.assertEqual(parsed["change_summary"], "baseline")
        self.assertEqual(parsed["hypothesis"], "test branch")

    def test_build_summary_recovers_description_fields_from_legacy_events(self):
        legacy_event = {
            "event_id": "evt-1",
            "event_type": search_memory.EVENT_TYPE_RUN,
            "recorded_at": "2026-03-22T09:00:00",
            "prepared_data_sha": "prep-1",
            "train_sha": "train-1",
            "experiment_description": (
                "move_intent=explore_new_branch | "
                "change_type=family_probe | "
                "family=LinearRegression | "
                "change=baseline | "
                "hypothesis=test branch"
            ),
            "status": run_experiment.STATUS_OK,
            "model_class": "LinearRegression",
            "model_params": {"fit_intercept": True},
            "candidate_signature": "cand-1",
            "model_signature": "model-1",
            "feature_signature": "feat-1",
            "feature_names": ["feature_a"],
            "n_features": 1,
            "n_base_features": 1,
            "n_derived_features": 0,
            "train_mape": 1.0,
            "val_mape": 2.0,
            "val_rmse": 3.0,
            "val_r2": 0.5,
            "train_val_mape_ratio": 2.0,
        }

        summary = search_memory.build_summary([legacy_event], "prep-1")

        self.assertEqual(
            summary["move_intent_distribution"],
            {"explore_new_branch": 1},
        )
        self.assertEqual(summary["best_run"]["move_intent"], "explore_new_branch")
        self.assertEqual(summary["best_run"]["change_type"], "family_probe")
        self.assertEqual(summary["best_run"]["declared_family"], "LinearRegression")
        self.assertEqual(summary["family_branch_depth"]["latest_move_intent"], "explore_new_branch")
        self.assertEqual(summary["consecutive_exploit_count"], 0)
        self.assertEqual(summary["family_loss_streaks"], {"LinearRegression": 0})
        self.assertEqual(summary["last_improvement_delta"], None)
        self.assertEqual(
            summary["plateau_signal"],
            {"delta": None, "is_plateau": False, "threshold": search_memory.PLATEAU_DELTA_THRESHOLD},
        )
        self.assertEqual(
            summary["overfit_signal"],
            {
                "is_overfit": True,
                "model_class": "LinearRegression",
                "threshold": search_memory.OVERFIT_RATIO_THRESHOLD,
                "train_sha": "train-1",
                "train_val_mape_ratio": 2.0,
            },
        )
        self.assertEqual(
            summary["recent_events"][0]["move_intent"],
            "explore_new_branch",
        )

    def test_build_summary_exposes_controller_signals(self):
        events = [
            {
                "event_id": "evt-1",
                "event_type": search_memory.EVENT_TYPE_RUN,
                "recorded_at": "2026-03-22T09:00:00",
                "prepared_data_sha": "prep-1",
                "train_sha": "train-1",
                "experiment_description": (
                    "move_intent=explore_new_branch | change_type=family_probe | "
                    "family=ExtraTrees | change=baseline"
                ),
                "status": run_experiment.STATUS_OK,
                "model_class": "ExtraTreesRegressor",
                "model_params": {"max_depth": 20},
                "candidate_signature": "cand-1",
                "model_signature": "model-1",
                "feature_signature": "feat-1",
                "feature_names": ["feature_a"],
                "n_features": 1,
                "n_base_features": 1,
                "n_derived_features": 0,
                "train_mape": 2.0,
                "val_mape": 6.0,
                "val_rmse": 3.0,
                "val_r2": 0.5,
                "train_val_mape_ratio": 1.2,
            },
            {
                "event_id": "evt-2",
                "event_type": search_memory.EVENT_TYPE_RUN,
                "recorded_at": "2026-03-22T09:05:00",
                "prepared_data_sha": "prep-1",
                "train_sha": "train-2",
                "experiment_description": (
                    "move_intent=exploit_current_winner | change_type=param_refine | "
                    "family=ExtraTrees | change=max_features 0.7->0.6"
                ),
                "status": run_experiment.STATUS_OK,
                "model_class": "ExtraTreesRegressor",
                "model_params": {"max_depth": 20, "max_features": 0.6},
                "candidate_signature": "cand-2",
                "model_signature": "model-2",
                "feature_signature": "feat-2",
                "feature_names": ["feature_a", "feature_b"],
                "n_features": 2,
                "n_base_features": 1,
                "n_derived_features": 1,
                "train_mape": 1.8,
                "val_mape": 6.05,
                "val_rmse": 3.1,
                "val_r2": 0.49,
                "train_val_mape_ratio": 1.6,
            },
            {
                "event_id": "evt-3",
                "event_type": search_memory.EVENT_TYPE_RUN,
                "recorded_at": "2026-03-22T09:10:00",
                "prepared_data_sha": "prep-1",
                "train_sha": "train-3",
                "experiment_description": (
                    "move_intent=exploit_current_winner | change_type=feature_cleanup | "
                    "family=ExtraTrees | change=drop noisy interaction"
                ),
                "status": run_experiment.STATUS_OK,
                "model_class": "ExtraTreesRegressor",
                "model_params": {"max_depth": 20, "max_features": 0.6},
                "candidate_signature": "cand-3",
                "model_signature": "model-3",
                "feature_signature": "feat-3",
                "feature_names": ["feature_a"],
                "n_features": 1,
                "n_base_features": 1,
                "n_derived_features": 0,
                "train_mape": 1.6,
                "val_mape": 6.05,
                "val_rmse": 3.05,
                "val_r2": 0.5,
                "train_val_mape_ratio": 1.7,
            },
        ]

        summary = search_memory.build_summary(events, "prep-1")

        self.assertEqual(summary["family_branch_depth"]["depth"], 3)
        self.assertEqual(summary["family_branch_depth"]["consecutive_exploit_count"], 2)
        self.assertEqual(summary["consecutive_exploit_count"], 2)
        self.assertEqual(summary["family_loss_streaks"], {"ExtraTreesRegressor": 2})
        self.assertEqual(summary["last_improvement_delta"], 0.05)
        self.assertEqual(
            summary["plateau_signal"],
            {"delta": 0.05, "is_plateau": True, "threshold": search_memory.PLATEAU_DELTA_THRESHOLD},
        )
        self.assertEqual(
            summary["overfit_signal"],
            {
                "is_overfit": True,
                "model_class": "ExtraTreesRegressor",
                "threshold": search_memory.OVERFIT_RATIO_THRESHOLD,
                "train_sha": "train-3",
                "train_val_mape_ratio": 1.7,
            },
        )
        self.assertEqual(summary["best_run"]["train_val_mape_ratio"], 1.2)
        self.assertEqual(summary["best_run"]["overfit_signal"], False)

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
        baseline_train_df, _test_df, meta = self.create_prepared_data()
        target_column = meta["target"]

        baseline = run_experiment.ensure_prepared_data_baseline()

        self.assertTrue(self.abs_path("experiments/session_baseline.json").exists())
        self.assertIn("prepared_data_sha", baseline)

        mutated_train = baseline_train_df.copy()
        mutated_train.loc[0, target_column] += 1.0
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
        self.assertEqual(events[0]["move_intent"], "explore_new_branch")
        self.assertEqual(events[0]["change_type"], "family_probe")
        self.assertEqual(events[0]["declared_family"], "LinearRegression")
        search_summary = self.load_search_summary()
        self.assertEqual(search_summary["counts"]["total_runs"], 1)
        self.assertEqual(search_summary["counts"]["successful_runs"], 1)
        self.assertEqual(
            search_summary["move_intent_distribution"],
            {"explore_new_branch": 1},
        )
        self.assertEqual(
            search_summary["family_branch_depth"]["model_class"],
            "LinearRegression",
        )
        self.assertEqual(search_summary["family_branch_depth"]["depth"], 1)

    def test_run_once_emits_top_feature_importances_for_tree_models(self):
        self.create_prepared_data()
        self.write_text("train.py", TREE_TRAIN_SOURCE)

        summary, exit_code = run_experiment.run_once()

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["status"], run_experiment.STATUS_OK)
        self.assertEqual(summary["model_class"], "RandomForestRegressor")
        self.assertEqual(summary["feature_importance_source"], "feature_importances_")
        self.assertEqual(summary["top_feature_importances"], {"x": 1.0})

        events = self.load_search_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["feature_importance_source"], "feature_importances_")
        self.assertEqual(events[0]["top_feature_importances"], {"x": 1.0})
        search_summary = self.load_search_summary()
        self.assertEqual(search_summary["best_run"]["feature_importance_source"], "feature_importances_")
        self.assertEqual(search_summary["best_run"]["top_feature_importances"], {"x": 1.0})
        self.assertEqual(
            search_summary["recent_events"][0]["top_feature_importances"],
            {"x": 1.0},
        )

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
        baseline_train_df, _test_df, meta = self.create_prepared_data()
        target_column = meta["target"]
        self.write_text("train.py", VALID_TRAIN_SOURCE)

        baseline_run, baseline_exit = run_experiment.run_once()
        self.assertEqual(baseline_exit, 0)
        self.assertEqual(len(self.load_search_events()), 1)

        mutated_train = baseline_train_df.copy()
        mutated_train.loc[0, target_column] += 10.0
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
        self.assertEqual(summary["move_intent_distribution"], {})
        self.assertIsNone(summary["family_branch_depth"])
        self.assertEqual(summary["consecutive_exploit_count"], 0)
        self.assertEqual(summary["family_loss_streaks"], {})
        self.assertIsNone(summary["overfit_signal"])
        self.assertIsNone(summary["last_improvement_delta"])
        self.assertEqual(
            summary["plateau_signal"],
            {"delta": None, "is_plateau": False, "threshold": search_memory.PLATEAU_DELTA_THRESHOLD},
        )
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
        self.assertEqual(
            search_summary["move_intent_distribution"],
            {"explore_new_branch": 2},
        )

    def test_search_summary_is_scoped_to_current_prepared_data_sha(self):
        baseline_train_df, _test_df, meta = self.create_prepared_data()
        target_column = meta["target"]
        self.write_text("train.py", VALID_TRAIN_SOURCE)

        first_summary, first_exit = run_experiment.run_once()
        self.assertEqual(first_exit, 0)
        first_scope = first_summary["prepared_data_sha"]

        self.abs_path("experiments/session_baseline.json").unlink()
        mutated_train = baseline_train_df.copy()
        mutated_train.loc[:, target_column] = mutated_train[target_column] + 25.0
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

    def test_workspace_specific_run_uses_isolated_paths(self):
        workspace = "workspaces/demo"
        self.create_prepared_data(prefix=workspace)
        self.write_text(f"{workspace}/train.py", VALID_TRAIN_SOURCE)

        summary, exit_code = run_experiment.run_once(workspace=workspace)

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["status"], run_experiment.STATUS_OK)
        self.assertEqual(summary["workspace_root"], workspace)
        self.assertTrue(self.abs_path(f"{workspace}/experiments/session_baseline.json").exists())
        self.assertTrue(self.abs_path(f"{workspace}/experiments/search_memory.jsonl").exists())
        self.assertFalse(self.abs_path("experiments/session_baseline.json").exists())
