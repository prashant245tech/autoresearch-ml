"""
run_experiment.py - Deterministic experiment runner for editable ML training specs.

This file owns:
- prepared-data loading
- deterministic train/validation splitting
- validation scoring under a fixed budget
- acceptance/export of a selected train.py variant
- local search-memory recording and summary retrieval

This file does not own:
- LLM calls
- git workflow
- autonomous search control
- best-model state
"""

import argparse
import ast
from datetime import datetime
import hashlib
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import shutil
import sys
import time
from types import ModuleType
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import search_memory
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from workspace_paths import WorkspacePaths

TRAIN_FILE_LABEL = "train.py"
DEFAULT_BASELINE_TRAIN_PATH = str(WorkspacePaths.from_value(".").baseline_train_path)
EXPERIMENT_BUDGET_S = 300
VALIDATION_SPLIT_SIZE = 0.2
VALIDATION_RANDOM_STATE = 42

STATUS_OK = "ok"
STATUS_BUDGET_EXCEEDED = "budget_exceeded"
STATUS_INVALID_CANDIDATE = "invalid_candidate"
STATUS_TRAIN_FAILED = "train_failed"
STATUS_HASH_MISMATCH = "hash_mismatch"
STATUS_PREPARED_DATA_MISMATCH = "prepared_data_mismatch"
STATUS_INIT_FAILED = "init_failed"

_ACTIVE_WORKSPACE_OVERRIDE: Optional[str] = None


class HarnessValidationError(Exception):
    status = STATUS_INVALID_CANDIDATE


class InvalidCandidateError(HarnessValidationError):
    status = STATUS_INVALID_CANDIDATE


class HashMismatchError(HarnessValidationError):
    status = STATUS_HASH_MISMATCH


class PreparedDataMismatchError(HarnessValidationError):
    status = STATUS_PREPARED_DATA_MISMATCH


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def configure_workspace(workspace: Optional[str] = None) -> WorkspacePaths:
    global _ACTIVE_WORKSPACE_OVERRIDE
    _ACTIVE_WORKSPACE_OVERRIDE = workspace
    return current_workspace_paths()


def current_workspace_paths() -> WorkspacePaths:
    return WorkspacePaths.from_value(_ACTIVE_WORKSPACE_OVERRIDE or ".")


def workspace_path_display(path) -> str:
    return current_workspace_paths().display_path(Path(path))


def make_output_dir(train_sha: str) -> str:
    paths = current_workspace_paths()
    return str(paths.accepted_models_dir / train_sha[:12])


def current_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value, key=repr)]
    if callable(value):
        return getattr(value, "__name__", repr(value))
    return repr(value)


def _baseline_payload(fingerprints: dict) -> dict:
    return {
        "created_at": current_timestamp(),
        "prepared_data": fingerprints,
    }


def load_prepared_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    paths = current_workspace_paths()
    for path in [paths.train_data_path, paths.test_data_path, paths.meta_path]:
        if not os.path.exists(path):
            raise RuntimeError(f"{path} not found. Run prepare.py first.")

    train_df = pd.read_parquet(paths.train_data_path)
    test_df = pd.read_parquet(paths.test_data_path)
    with open(paths.meta_path) as handle:
        meta = json.load(handle)
    return train_df, test_df, meta


def resolve_target_column(meta: dict) -> str:
    paths = current_workspace_paths()
    target_column = meta.get("target")
    if not isinstance(target_column, str) or not target_column.strip():
        raise InvalidCandidateError(
            f"Prepared metadata in {paths.display_path(paths.meta_path)} "
            "must define a non-empty `target` field."
        )
    return target_column


def split_features_and_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, np.ndarray]:
    if target_column not in df.columns:
        raise InvalidCandidateError(
            f"Prepared dataset is missing target column `{target_column}`."
        )
    X = df.drop(columns=[target_column])
    y = df[target_column].to_numpy()
    return X, y


def prepared_data_fingerprints() -> dict:
    paths = current_workspace_paths()
    fingerprints = {
        "train_data_sha": sha256_file(str(paths.train_data_path)),
        "test_data_sha": sha256_file(str(paths.test_data_path)),
        "meta_sha": sha256_file(str(paths.meta_path)),
    }
    fingerprints["prepared_data_sha"] = sha256_text(
        json.dumps(fingerprints, sort_keys=True)
    )
    return fingerprints


def ensure_prepared_data_baseline() -> dict:
    paths = current_workspace_paths()
    fingerprints = prepared_data_fingerprints()
    os.makedirs(paths.experiments_dir, exist_ok=True)

    if not os.path.exists(paths.session_baseline_path):
        with open(paths.session_baseline_path, "w") as handle:
            json.dump(_baseline_payload(fingerprints), handle, indent=2, sort_keys=True)
        return fingerprints

    with open(paths.session_baseline_path) as handle:
        baseline = json.load(handle)

    baseline_fingerprints = baseline.get("prepared_data", {})
    mismatch_keys = [
        key
        for key, current_value in fingerprints.items()
        if baseline_fingerprints.get(key) != current_value
    ]
    if mismatch_keys:
        mismatch_lines = ", ".join(mismatch_keys)
        raise PreparedDataMismatchError(
            "Prepared data changed since the session baseline "
            f"({mismatch_lines}). Delete {paths.display_path(paths.session_baseline_path)} "
            "after rerunning prepare.py "
            "to start a new tuning session."
        )

    return fingerprints


def current_prepared_data_sha_or_none() -> Optional[str]:
    paths = current_workspace_paths()
    if not all(
        os.path.exists(path)
        for path in [paths.train_data_path, paths.test_data_path, paths.meta_path]
    ):
        return None
    return prepared_data_fingerprints().get("prepared_data_sha")


def split_train_for_validation(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_train_df, val_df = train_test_split(
        train_df,
        test_size=VALIDATION_SPLIT_SIZE,
        random_state=VALIDATION_RANDOM_STATE,
        shuffle=True,
    )
    return fit_train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def extract_description_from_source(train_code: str) -> str:
    try:
        tree = ast.parse(train_code, filename=TRAIN_FILE_LABEL)
    except SyntaxError:
        return "experiment description unavailable"

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "EXPERIMENT_DESCRIPTION":
                try:
                    value = ast.literal_eval(node.value)
                except Exception:
                    return "experiment description unavailable"
                if isinstance(value, str) and value.strip():
                    return " ".join(value.split())
    return "experiment description unavailable"


def _validate_callable_signature(fn, name: str, expected_args: list[str]) -> None:
    signature = inspect.signature(fn)
    actual_args = [
        param.name
        for param in signature.parameters.values()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if actual_args[: len(expected_args)] != expected_args:
        raise InvalidCandidateError(
            f"{name} must keep the signature {name}({', '.join(expected_args)})."
        )


def validate_train_module(module) -> None:
    description = getattr(module, "EXPERIMENT_DESCRIPTION", None)
    if not isinstance(description, str) or not description.strip():
        raise InvalidCandidateError("train.py must define a non-empty EXPERIMENT_DESCRIPTION.")

    engineer_features = getattr(module, "engineer_features", None)
    if not callable(engineer_features):
        raise InvalidCandidateError("train.py must define engineer_features(df, meta).")
    _validate_callable_signature(engineer_features, "engineer_features", ["df", "meta"])

    build_model = getattr(module, "build_model", None)
    if not callable(build_model):
        raise InvalidCandidateError("train.py must define build_model(meta).")
    _validate_callable_signature(build_model, "build_model", ["meta"])

    fit_model = getattr(module, "fit_model", None)
    if fit_model is not None:
        if not callable(fit_model):
            raise InvalidCandidateError("fit_model must be callable when defined.")
        _validate_callable_signature(
            fit_model,
            "fit_model",
            ["model", "X_train", "y_train", "X_val", "y_val"],
        )

    predict_model = getattr(module, "predict_model", None)
    if predict_model is not None:
        if not callable(predict_model):
            raise InvalidCandidateError("predict_model must be callable when defined.")
        _validate_callable_signature(predict_model, "predict_model", ["model", "X"])


def load_train_spec_from_source(train_source: str):
    try:
        ast.parse(train_source, filename=TRAIN_FILE_LABEL)
    except SyntaxError as exc:
        raise InvalidCandidateError(f"train.py is not valid Python: {exc}") from exc

    module_name = f"train_runtime_{int(time.time() * 1_000_000)}"
    module = ModuleType(module_name)
    module.__file__ = TRAIN_FILE_LABEL
    module.__dict__["__name__"] = module_name
    exec(compile(train_source, TRAIN_FILE_LABEL, "exec"), module.__dict__)
    validate_train_module(module)
    return module


def load_train_spec():
    paths = current_workspace_paths()
    if not os.path.exists(paths.train_py_path):
        init_command = "python run_experiment.py init-train"
        if paths.display_workspace_root() != ".":
            init_command += f" --workspace {paths.display_workspace_root()}"
        raise InvalidCandidateError(
            f"{paths.display_path(paths.train_py_path)} not found. "
            f"Run `{init_command}` "
            f"to create it from {paths.display_path(paths.baseline_train_path)}."
        )
    with open(paths.train_py_path) as handle:
        train_source = handle.read()
    train_sha = sha256_text(train_source)
    description = extract_description_from_source(train_source)
    train_spec = load_train_spec_from_source(train_source)
    return train_spec, train_source, train_sha, description


def remaining_budget(start_time: float) -> float:
    return EXPERIMENT_BUDGET_S - (time.monotonic() - start_time)


def build_model_from_spec(train_spec, meta: dict):
    model = train_spec.build_model(meta)
    if model is None:
        raise InvalidCandidateError("build_model(meta) returned None.")
    return model


def fit_model_from_spec(train_spec, model, X_train, y_train, X_val, y_val):
    fit_model = getattr(train_spec, "fit_model", None)
    if fit_model is None:
        fitted_model = model.fit(X_train, y_train)
    else:
        fitted_model = fit_model(model, X_train, y_train, X_val, y_val)
    return fitted_model if fitted_model is not None else model


def predict_with_spec(train_spec, model, X: pd.DataFrame) -> np.ndarray:
    predict_model = getattr(train_spec, "predict_model", None)
    if predict_model is None:
        y_pred = model.predict(X)
    else:
        y_pred = predict_model(model, X)

    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_pred) != len(X):
        raise InvalidCandidateError(
            f"predict_model() returned {len(y_pred)} predictions for {len(X)} rows."
        )
    return np.clip(y_pred, 0, None)


def engineer_numeric_features(train_spec, raw_df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    feature_df = train_spec.engineer_features(raw_df, meta)
    if not isinstance(feature_df, pd.DataFrame):
        raise InvalidCandidateError("engineer_features() must return a pandas DataFrame.")

    numeric_df = feature_df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        raise InvalidCandidateError("engineer_features() returned no numeric features.")
    return numeric_df


def align_feature_frame(feature_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    return feature_df.reindex(columns=feature_names, fill_value=0)


def summarize_feature_telemetry(feature_names: list[str], meta: dict) -> dict:
    base_feature_candidates = list(meta.get("model_features", []))
    base_feature_set = set(base_feature_candidates)
    base_feature_names = [name for name in feature_names if name in base_feature_set]
    derived_feature_names = [name for name in feature_names if name not in base_feature_set]
    omitted_base_feature_names = [
        name for name in base_feature_candidates if name not in set(base_feature_names)
    ]
    return {
        "base_feature_names": base_feature_names,
        "derived_feature_names": derived_feature_names,
        "n_base_features": len(base_feature_names),
        "n_derived_features": len(derived_feature_names),
        "n_candidate_base_features": len(base_feature_candidates),
        "omitted_base_feature_names": omitted_base_feature_names,
    }


def extract_model_params(model) -> Optional[dict]:
    get_params = getattr(model, "get_params", None)
    if not callable(get_params):
        return None
    try:
        params = get_params(deep=False)
    except TypeError:
        params = get_params()
    if not isinstance(params, dict):
        return None
    return {str(key): json_safe(value) for key, value in params.items()}


def extract_top_feature_importances(
    model,
    feature_names: list[str],
    limit: int = 15,
) -> tuple[Optional[str], Optional[dict]]:
    raw_importances = getattr(model, "feature_importances_", None)
    if raw_importances is None:
        return None, None

    importances = np.asarray(raw_importances, dtype=float).reshape(-1)
    if len(importances) != len(feature_names):
        return None, None

    ranked_pairs = sorted(
        zip(feature_names, importances),
        key=lambda item: (-float(item[1]), item[0]),
    )
    top_pairs = ranked_pairs[:limit]
    top_feature_importances = {
        feature_name: round(float(score), 6)
        for feature_name, score in top_pairs
    }
    return "feature_importances_", top_feature_importances


def build_generalization_telemetry(train_metrics: dict, val_metrics: dict) -> dict:
    train_mape = train_metrics["train_mape"]
    val_mape = val_metrics["val_mape"]
    train_rmse = train_metrics["train_rmse"]
    val_rmse = val_metrics["val_rmse"]
    return {
        "train_val_mape_gap": round(float(val_mape - train_mape), 4),
        "train_val_rmse_gap": round(float(val_rmse - train_rmse), 4),
        "train_val_mape_ratio": round(float(val_mape / max(train_mape, 1e-9)), 4),
    }


def prepare_validation_inputs(train_spec, train_df: pd.DataFrame, meta: dict):
    fit_train_df, val_df = split_train_for_validation(train_df)
    target_column = resolve_target_column(meta)
    X_train_raw, y_train = split_features_and_target(fit_train_df, target_column)
    X_val_raw, y_val = split_features_and_target(val_df, target_column)

    X_train = engineer_numeric_features(train_spec, X_train_raw, meta)
    feature_names = list(X_train.columns)
    X_val = engineer_numeric_features(train_spec, X_val_raw, meta)
    X_val = align_feature_frame(X_val, feature_names)

    return X_train, y_train, X_val, y_val, feature_names


def prepare_full_train_inputs(train_spec, train_df: pd.DataFrame, meta: dict):
    target_column = resolve_target_column(meta)
    X_train_raw, y_train = split_features_and_target(train_df, target_column)
    X_train = engineer_numeric_features(train_spec, X_train_raw, meta)
    feature_names = list(X_train.columns)
    return X_train, y_train, feature_names


def prepare_test_inputs(train_spec, test_df: pd.DataFrame, meta: dict, feature_names: list[str]):
    target_column = resolve_target_column(meta)
    X_test_raw, y_test = split_features_and_target(test_df, target_column)
    X_test = engineer_numeric_features(train_spec, X_test_raw, meta)
    X_test = align_feature_frame(X_test, feature_names)
    return X_test, y_test


def build_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_mape": round(float(mean_absolute_percentage_error(y_true, y_pred) * 100), 4),
        f"{prefix}_rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        f"{prefix}_r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def _fit_and_score_worker(queue, train_source, X_train, y_train, X_val, y_val, meta):
    try:
        train_spec = load_train_spec_from_source(train_source)
        model = build_model_from_spec(train_spec, meta)
        model_params = extract_model_params(model)

        fit_started = time.monotonic()
        fitted_model = fit_model_from_spec(train_spec, model, X_train, y_train, X_val, y_val)
        fit_elapsed_s = round(float(time.monotonic() - fit_started), 4)
        if not hasattr(fitted_model, "predict"):
            raise InvalidCandidateError("Fitted model must expose predict(X).")
        feature_importance_source, top_feature_importances = extract_top_feature_importances(
            fitted_model,
            list(X_train.columns),
        )

        predict_train_started = time.monotonic()
        y_train_pred = predict_with_spec(train_spec, fitted_model, X_train)
        predict_train_elapsed_s = round(float(time.monotonic() - predict_train_started), 4)

        predict_val_started = time.monotonic()
        y_val_pred = predict_with_spec(train_spec, fitted_model, X_val)
        predict_val_elapsed_s = round(float(time.monotonic() - predict_val_started), 4)

        train_metrics = build_metrics(y_train, y_train_pred, "train")
        val_metrics = build_metrics(y_val, y_val_pred, "val")

        result = {
            "fit_elapsed_s": fit_elapsed_s,
            "status": STATUS_OK,
            "model_class": type(fitted_model).__name__,
            "model_params": model_params,
            "predict_train_elapsed_s": predict_train_elapsed_s,
            "predict_val_elapsed_s": predict_val_elapsed_s,
            **train_metrics,
            **val_metrics,
            **build_generalization_telemetry(train_metrics, val_metrics),
        }
        if feature_importance_source and top_feature_importances is not None:
            result["feature_importance_source"] = feature_importance_source
            result["top_feature_importances"] = top_feature_importances

        queue.put(result)
    except HarnessValidationError as exc:
        queue.put({"status": exc.status, "error": str(exc)})
    except Exception as exc:
        queue.put({"status": STATUS_TRAIN_FAILED, "error": str(exc)})


def fit_and_score_with_budget(
    train_source: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    meta: dict,
    start_time: float,
) -> dict:
    budget_left = remaining_budget(start_time)
    if budget_left <= 0:
        return {
            "status": STATUS_BUDGET_EXCEEDED,
            "error": "Budget exhausted before model fitting started.",
        }

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_fit_and_score_worker,
        args=(queue, train_source, X_train, y_train, X_val, y_val, meta),
    )
    process.start()
    process.join(timeout=budget_left)

    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        queue.close()
        queue.join_thread()
        return {
            "status": STATUS_BUDGET_EXCEEDED,
            "error": f"Experiment exceeded the fixed {EXPERIMENT_BUDGET_S}s budget.",
        }

    try:
        result = queue.get(timeout=1)
    except Empty:
        result = {
            "status": STATUS_TRAIN_FAILED,
            "error": "Training process exited without returning results.",
        }
    finally:
        queue.close()
        queue.join_thread()

    return result


def record_search_memory(event_type: str, payload: dict) -> Optional[dict]:
    paths = current_workspace_paths()
    return search_memory.record_event(
        event_type,
        payload,
        events_path=str(paths.search_memory_path),
        summary_path=str(paths.search_summary_path),
    )


def run_once(workspace: Optional[str] = None) -> tuple[dict, int]:
    configure_workspace(workspace)
    start_time = time.monotonic()
    paths = current_workspace_paths()

    try:
        train_spec, train_source, train_sha, description = load_train_spec()
        fingerprints = ensure_prepared_data_baseline()
        train_df, _test_df, meta = load_prepared_data()

        X_train, y_train, X_val, y_val, feature_names = prepare_validation_inputs(
            train_spec, train_df, meta
        )
        feature_telemetry = summarize_feature_telemetry(feature_names, meta)
        build_model_from_spec(train_spec, meta)

        worker_result = fit_and_score_with_budget(
            train_source,
            X_train,
            y_train,
            X_val,
            y_val,
            meta,
            start_time,
        )

        summary = {
            "budget_s": EXPERIMENT_BUDGET_S,
            "elapsed_s": round(float(time.monotonic() - start_time), 1),
            "experiment_description": description,
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "n_train_rows": int(len(X_train)),
            "n_val_rows": int(len(X_val)),
            "session_baseline_path": paths.display_path(paths.session_baseline_path),
            "status": worker_result["status"],
            "train_sha": train_sha,
            "validation_random_state": VALIDATION_RANDOM_STATE,
            "validation_split_size": VALIDATION_SPLIT_SIZE,
            "workspace_root": paths.display_workspace_root(),
            **fingerprints,
            **feature_telemetry,
        }
        summary["meta_version"] = meta.get("version")
        summary["elapsed_budget_fraction"] = round(
            float(summary["elapsed_s"] / EXPERIMENT_BUDGET_S), 4
        )

        if worker_result["status"] == STATUS_OK:
            summary["model_class"] = worker_result["model_class"]
            summary["model_params"] = worker_result["model_params"]
            if "feature_importance_source" in worker_result:
                summary["feature_importance_source"] = worker_result["feature_importance_source"]
            if "top_feature_importances" in worker_result:
                summary["top_feature_importances"] = worker_result["top_feature_importances"]
            summary["fit_elapsed_s"] = worker_result["fit_elapsed_s"]
            summary["predict_train_elapsed_s"] = worker_result["predict_train_elapsed_s"]
            summary["predict_val_elapsed_s"] = worker_result["predict_val_elapsed_s"]
            summary["train_mape"] = worker_result["train_mape"]
            summary["train_rmse"] = worker_result["train_rmse"]
            summary["train_r2"] = worker_result["train_r2"]
            summary["val_mape"] = worker_result["val_mape"]
            summary["val_rmse"] = worker_result["val_rmse"]
            summary["val_r2"] = worker_result["val_r2"]
            summary["train_val_mape_gap"] = worker_result["train_val_mape_gap"]
            summary["train_val_rmse_gap"] = worker_result["train_val_rmse_gap"]
            summary["train_val_mape_ratio"] = worker_result["train_val_mape_ratio"]
            record_search_memory(search_memory.EVENT_TYPE_RUN, summary)
            return summary, 0

        summary["error"] = worker_result.get("error")
        record_search_memory(search_memory.EVENT_TYPE_RUN, summary)
        return summary, 1
    except Exception as exc:
        failure = {
            "budget_s": EXPERIMENT_BUDGET_S,
            "elapsed_s": round(float(time.monotonic() - start_time), 1),
            "experiment_description": extract_description_from_source(
                paths.train_py_path.read_text() if paths.train_py_path.exists() else ""
            ),
            "status": getattr(exc, "status", STATUS_TRAIN_FAILED),
            "error": str(exc),
            "workspace_root": paths.display_workspace_root(),
        }
        if paths.train_py_path.exists():
            failure["train_sha"] = sha256_file(str(paths.train_py_path))
        if all(
            os.path.exists(path)
            for path in [paths.train_data_path, paths.test_data_path, paths.meta_path]
        ):
            failure.update(prepared_data_fingerprints())
        failure["session_baseline_path"] = paths.display_path(paths.session_baseline_path)
        record_search_memory(search_memory.EVENT_TYPE_RUN, failure)
        return failure, 1


def accept_once(
    expected_train_sha: str,
    output_dir: Optional[str] = None,
    workspace: Optional[str] = None,
) -> tuple[dict, int]:
    configure_workspace(workspace)
    start_time = time.monotonic()
    output_path: Optional[Path] = None
    paths = current_workspace_paths()

    try:
        train_spec, _train_source, train_sha, description = load_train_spec()
        if train_sha != expected_train_sha:
            raise HashMismatchError(
                f"Current train.py hash {train_sha} does not match expected {expected_train_sha}."
            )

        fingerprints = ensure_prepared_data_baseline()
        train_df, test_df, meta = load_prepared_data()

        X_train, y_train, feature_names = prepare_full_train_inputs(train_spec, train_df, meta)
        X_test, y_test = prepare_test_inputs(train_spec, test_df, meta, feature_names)
        feature_telemetry = summarize_feature_telemetry(feature_names, meta)

        model = build_model_from_spec(train_spec, meta)
        model_params = extract_model_params(model)
        fit_started = time.monotonic()
        fitted_model = fit_model_from_spec(train_spec, model, X_train, y_train, X_train, y_train)
        fit_elapsed_s = round(float(time.monotonic() - fit_started), 4)
        if not hasattr(fitted_model, "predict"):
            raise InvalidCandidateError("Fitted model must expose predict(X).")

        predict_test_started = time.monotonic()
        y_test_pred = predict_with_spec(train_spec, fitted_model, X_test)
        predict_test_elapsed_s = round(float(time.monotonic() - predict_test_started), 4)
        test_metrics = build_metrics(y_test, y_test_pred, "test")

        resolved_output_dir = Path(output_dir) if output_dir else Path(make_output_dir(train_sha))
        if not resolved_output_dir.is_absolute():
            resolved_output_dir = paths.workspace_root / resolved_output_dir
        output_path = resolved_output_dir.resolve()
        if output_path.exists():
            raise RuntimeError(
                f"Artifact directory already exists: {paths.display_path(output_path)}. "
                "Choose a new --output-dir."
            )

        output_path.mkdir(parents=True, exist_ok=False)

        model_path = output_path / "model.pkl"
        features_path = output_path / "feature_columns.json"
        train_copy_path = output_path / "train.py"
        manifest_path = output_path / "manifest.json"

        joblib.dump(fitted_model, model_path)
        with open(features_path, "w") as handle:
            json.dump(feature_names, handle, indent=2)
        shutil.copyfile(paths.train_py_path, train_copy_path)

        manifest = {
            "artifact_paths": {
                "feature_columns": str(features_path),
                "manifest": str(manifest_path),
                "model": str(model_path),
                "train_py": str(train_copy_path),
            },
            "elapsed_s": round(float(time.monotonic() - start_time), 1),
            "experiment_description": description,
            "feature_names": feature_names,
            "fit_elapsed_s": fit_elapsed_s,
            "meta_version": meta.get("version"),
            "model_class": type(fitted_model).__name__,
            "model_params": model_params,
            "n_features": len(feature_names),
            "n_test_rows": int(len(X_test)),
            "n_train_rows": int(len(X_train)),
            "output_dir": str(output_path),
            "predict_test_elapsed_s": predict_test_elapsed_s,
            "saved_at": current_timestamp(),
            "session_baseline_path": paths.display_path(paths.session_baseline_path),
            "status": STATUS_OK,
            "train_sha": train_sha,
            "workspace_root": paths.display_workspace_root(),
            **fingerprints,
            **feature_telemetry,
            **test_metrics,
        }

        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

        record_search_memory(search_memory.EVENT_TYPE_ACCEPT, manifest)
        return manifest, 0
    except Exception as exc:
        if output_path is not None and output_path.exists():
            shutil.rmtree(output_path, ignore_errors=True)
        failure = {
            "elapsed_s": round(float(time.monotonic() - start_time), 1),
            "experiment_description": extract_description_from_source(
                paths.train_py_path.read_text() if paths.train_py_path.exists() else ""
            ),
            "status": getattr(exc, "status", STATUS_TRAIN_FAILED),
            "error": str(exc),
            "expected_train_sha": expected_train_sha,
            "workspace_root": paths.display_workspace_root(),
        }
        if paths.train_py_path.exists():
            failure["train_sha"] = sha256_file(str(paths.train_py_path))
        if all(
            os.path.exists(path)
            for path in [paths.train_data_path, paths.test_data_path, paths.meta_path]
        ):
            failure.update(prepared_data_fingerprints())
        failure["session_baseline_path"] = paths.display_path(paths.session_baseline_path)
        record_search_memory(search_memory.EVENT_TYPE_ACCEPT, failure)
        return failure, 1


def memory_summary_once(workspace: Optional[str] = None) -> tuple[dict, int]:
    configure_workspace(workspace)
    paths = current_workspace_paths()
    try:
        summary = search_memory.get_or_rebuild_summary(
            prepared_data_sha=current_prepared_data_sha_or_none(),
            events_path=str(paths.search_memory_path),
            summary_path=str(paths.search_summary_path),
        )
        return summary, 0
    except Exception as exc:
        return (
            {
                "status": STATUS_TRAIN_FAILED,
                "error": str(exc),
                "search_memory_path": paths.display_path(paths.search_memory_path),
                "search_summary_path": paths.display_path(paths.search_summary_path),
                "prepared_data_sha": current_prepared_data_sha_or_none(),
                "workspace_root": paths.display_workspace_root(),
            },
            1,
        )


def init_train_once(
    force: bool = False,
    baseline_path: Optional[str] = None,
    workspace: Optional[str] = None,
) -> tuple[dict, int]:
    configure_workspace(workspace)
    paths = current_workspace_paths()
    source_path = Path(baseline_path) if baseline_path else paths.baseline_train_path
    if not source_path.is_absolute():
        source_path = (Path.cwd() / source_path).resolve()

    try:
        if not os.path.exists(source_path):
            raise RuntimeError(
                f"Baseline train spec not found: {source_path}."
            )
        if paths.train_py_path.exists() and not force:
            raise RuntimeError(
                f"{paths.display_path(paths.train_py_path)} already exists. "
                "Re-run with `--force` to replace it."
            )

        paths.train_py_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, paths.train_py_path)
        with open(paths.train_py_path, encoding="utf-8") as handle:
            train_source = handle.read()

        return (
            {
                "status": STATUS_OK,
                "baseline_path": paths.display_path(source_path),
                "experiment_description": extract_description_from_source(train_source),
                "message": (
                    f"Created local {paths.display_path(paths.train_py_path)} "
                    f"from {paths.display_path(source_path)}. "
                    "Edit train.py and then run "
                    f"`python run_experiment.py run"
                    + (
                        f" --workspace {paths.display_workspace_root()}`."
                        if paths.display_workspace_root() != "."
                        else "`."
                    )
                ),
                "train_path": paths.display_path(paths.train_py_path),
                "train_sha": sha256_text(train_source),
                "workspace_root": paths.display_workspace_root(),
            },
            0,
        )
    except Exception as exc:
        return (
            {
                "status": STATUS_INIT_FAILED,
                "baseline_path": paths.display_path(source_path),
                "error": str(exc),
                "train_path": paths.display_path(paths.train_py_path),
                "workspace_root": paths.display_workspace_root(),
            },
            1,
        )


def run_single_from_train_entrypoint() -> None:
    summary, exit_code = run_once()
    print_json(summary)
    if exit_code != 0:
        sys.exit(exit_code)


def add_workspace_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace",
        default=".",
        help=(
            "Workspace root containing local config/data/train.py state "
            "(default: current directory)."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic AutoResearch runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init-train",
        help="Create a local editable train.py from a tracked baseline",
    )
    add_workspace_argument(init_parser)
    init_parser.add_argument(
        "--baseline-path",
        default=DEFAULT_BASELINE_TRAIN_PATH,
        help="Source baseline to copy into the workspace train.py.",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing workspace train.py.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run validation-only evaluation for the current workspace train.py",
    )
    add_workspace_argument(run_parser)

    summary_parser = subparsers.add_parser(
        "memory-summary",
        help="Print the current prepared-data-scoped search-memory summary",
    )
    add_workspace_argument(summary_parser)

    accept_parser = subparsers.add_parser(
        "accept",
        help="Retrain the accepted workspace train.py on all train data and save artifact bundle",
    )
    add_workspace_argument(accept_parser)
    accept_parser.add_argument(
        "--expected-train-sha",
        required=True,
        help="train_sha reported by `python run_experiment.py run`.",
    )
    accept_parser.add_argument(
        "--output-dir",
        help="Override the default artifact directory (defaults to models/accepted/<train_sha[:12]>).",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init-train":
        summary, exit_code = init_train_once(args.force, args.baseline_path, args.workspace)
    elif args.command == "run":
        summary, exit_code = run_once(args.workspace)
    elif args.command == "memory-summary":
        summary, exit_code = memory_summary_once(args.workspace)
    else:
        summary, exit_code = accept_once(
            args.expected_train_sha,
            args.output_dir,
            args.workspace,
        )

    print_json(summary)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
