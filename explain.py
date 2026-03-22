"""
explain.py - Optional SHAP post-hoc analysis for accepted artifacts.

This file owns:
- loading accepted artifact bundles
- reconstructing feature matrices using the accepted train.py
- running optional SHAP tree explanations on train/test data
- writing a compact JSON explanation summary

This file does not own:
- acceptance decisions
- search control
- artifact creation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

import run_experiment

DEFAULT_DATASET = "test"
DEFAULT_SAMPLE_SIZE = 256
DEFAULT_TOP_K = 15
DEFAULT_RANDOM_STATE = 42

STATUS_OK = "ok"
STATUS_EXPLAIN_FAILED = "explain_failed"
STATUS_INVALID_ARTIFACT = "invalid_artifact"
STATUS_MISSING_DEPENDENCY = "missing_dependency"
STATUS_PREPARED_DATA_MISMATCH = "prepared_data_mismatch"


class ExplainError(Exception):
    status = STATUS_INVALID_ARTIFACT


class InvalidArtifactError(ExplainError):
    status = STATUS_INVALID_ARTIFACT


class MissingDependencyError(ExplainError):
    status = STATUS_MISSING_DEPENDENCY


class PreparedDataMismatchError(ExplainError):
    status = STATUS_PREPARED_DATA_MISMATCH


def print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def load_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_artifact_bundle(artifact_dir: str) -> dict:
    artifact_path = Path(artifact_dir)
    manifest_path = artifact_path / "manifest.json"
    if not manifest_path.exists():
        raise InvalidArtifactError(f"Artifact manifest not found: {manifest_path}")

    manifest = load_json(manifest_path)
    artifact_paths = manifest.get("artifact_paths", {})

    model_path = Path(artifact_paths.get("model", artifact_path / "model.pkl"))
    features_path = Path(
        artifact_paths.get("feature_columns", artifact_path / "feature_columns.json")
    )
    train_py_path = Path(artifact_paths.get("train_py", artifact_path / "train.py"))

    for path in [model_path, features_path, train_py_path]:
        if not path.exists():
            raise InvalidArtifactError(f"Artifact bundle is incomplete; missing {path}")

    return {
        "artifact_dir": artifact_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "model_path": model_path,
        "features_path": features_path,
        "train_py_path": train_py_path,
    }


def default_output_path(artifact_dir: str) -> Path:
    return Path(artifact_dir) / "explanations" / "shap_summary.json"


def load_feature_names(path: Path) -> list[str]:
    payload = load_json(path)
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise InvalidArtifactError(f"feature_columns.json must contain a JSON list of strings: {path}")
    return payload


def load_accepted_train_spec(train_py_path: Path):
    train_source = train_py_path.read_text(encoding="utf-8")
    train_spec = run_experiment.load_train_spec_from_source(train_source)
    description = run_experiment.extract_description_from_source(train_source)
    train_sha = run_experiment.sha256_text(train_source)
    return train_spec, description, train_sha


def sample_dataset(df: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    if sample_size <= 0:
        raise InvalidArtifactError("--sample-size must be a positive integer.")
    if len(df) <= sample_size:
        return df.reset_index(drop=True)
    return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def resolve_target_column(meta: dict) -> str:
    try:
        return run_experiment.resolve_target_column(meta)
    except run_experiment.InvalidCandidateError as exc:
        raise InvalidArtifactError(str(exc)) from exc


def build_feature_frame(
    train_spec,
    df: pd.DataFrame,
    meta: dict,
    feature_names: list[str],
) -> pd.DataFrame:
    target_column = resolve_target_column(meta)
    raw_df = df.drop(columns=[target_column]) if target_column in df.columns else df.copy()
    feature_df = run_experiment.engineer_numeric_features(train_spec, raw_df, meta)
    return run_experiment.align_feature_frame(feature_df, feature_names)


def import_shap_module():
    try:
        import shap
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "SHAP is not installed. Install it with `python3 -m pip install shap` "
            "to use explain.py."
        ) from exc
    return shap


def normalize_shap_values(shap_values) -> np.ndarray:
    if isinstance(shap_values, list):
        if len(shap_values) != 1:
            raise InvalidArtifactError(
                "Unsupported SHAP output: multi-output shap_values are not supported yet."
            )
        shap_values = shap_values[0]

    if hasattr(shap_values, "values"):
        array = np.asarray(shap_values.values, dtype=float)
    else:
        array = np.asarray(shap_values, dtype=float)

    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise InvalidArtifactError(
            f"Unsupported SHAP output shape {array.shape}; expected 2-D values."
        )
    return array


def summarize_shap_values(
    model,
    X: pd.DataFrame,
    top_k: int,
) -> dict:
    shap = import_shap_module()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    array = normalize_shap_values(shap_values)

    if array.shape[1] != len(X.columns):
        raise InvalidArtifactError(
            "SHAP output width does not match feature columns for the explanation dataset."
        )

    mean_abs = np.abs(array).mean(axis=0)
    ranked_pairs = sorted(
        zip(list(X.columns), mean_abs),
        key=lambda item: (-float(item[1]), item[0]),
    )
    top_pairs = ranked_pairs[:top_k]
    expected_value = getattr(explainer, "expected_value", None)
    expected_value = run_experiment.json_safe(expected_value)

    return {
        "shap_backend": "TreeExplainer",
        "shap_expected_value": expected_value,
        "top_mean_abs_shap": {
            feature_name: round(float(score), 6)
            for feature_name, score in top_pairs
        },
    }


def resolve_dataset(dataset: str, meta: dict) -> tuple[pd.DataFrame, str, str]:
    paths = run_experiment.current_workspace_paths()
    train_df, test_df, _meta = run_experiment.load_prepared_data()
    target_column = resolve_target_column(meta)
    if dataset == "train":
        return train_df, paths.display_path(paths.train_data_path), target_column
    if dataset == "test":
        return test_df, paths.display_path(paths.test_data_path), target_column
    raise InvalidArtifactError(f"Unsupported dataset split: {dataset}")


def resolve_artifact_dir(artifact_dir: str) -> str:
    candidate = Path(artifact_dir)
    if candidate.is_absolute():
        return str(candidate)
    return str((run_experiment.current_workspace_paths().workspace_root / candidate).resolve())


def build_explanation_summary(
    artifact_dir: str,
    dataset: str,
    sample_size: int,
    top_k: int,
    output_path: Optional[str] = None,
    allow_prepared_data_mismatch: bool = False,
    workspace: Optional[str] = None,
) -> tuple[dict, int]:
    run_experiment.configure_workspace(workspace)
    try:
        bundle = load_artifact_bundle(resolve_artifact_dir(artifact_dir))
        manifest = bundle["manifest"]
        current_prepared_data_sha = run_experiment.current_prepared_data_sha_or_none()
        expected_prepared_data_sha = manifest.get("prepared_data_sha")
        prepared_data_match = (
            current_prepared_data_sha == expected_prepared_data_sha
            if current_prepared_data_sha and expected_prepared_data_sha
            else None
        )
        if (
            not allow_prepared_data_mismatch
            and current_prepared_data_sha
            and expected_prepared_data_sha
            and current_prepared_data_sha != expected_prepared_data_sha
        ):
            raise PreparedDataMismatchError(
                "Current prepared data does not match the accepted artifact. "
                "Restore the matching prepared data or rerun with "
                "--allow-prepared-data-mismatch if you intentionally want to proceed."
            )

        model = joblib.load(bundle["model_path"])
        feature_names = load_feature_names(bundle["features_path"])
        train_spec, description, accepted_train_sha = load_accepted_train_spec(
            bundle["train_py_path"]
        )
        _train_df, _test_df, meta = run_experiment.load_prepared_data()
        df, dataset_path, target_column = resolve_dataset(dataset, meta)
        sampled_df = sample_dataset(df, sample_size, DEFAULT_RANDOM_STATE)
        X = build_feature_frame(train_spec, sampled_df, meta, feature_names)
        shap_summary = summarize_shap_values(model, X, top_k)

        resolved_output_path = Path(output_path) if output_path else default_output_path(artifact_dir)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "status": STATUS_OK,
            "artifact_dir": str(bundle["artifact_dir"]),
            "manifest_path": str(bundle["manifest_path"]),
            "output_path": str(resolved_output_path),
            "dataset": dataset,
            "dataset_path": dataset_path,
            "target_column": target_column,
            "sample_size_requested": sample_size,
            "sample_size_used": int(len(sampled_df)),
            "top_k": top_k,
            "train_sha": manifest.get("train_sha") or accepted_train_sha,
            "experiment_description": description,
            "model_class": manifest.get("model_class") or type(model).__name__,
            "prepared_data_sha": expected_prepared_data_sha,
            "current_prepared_data_sha": current_prepared_data_sha,
            "prepared_data_match": prepared_data_match,
            "feature_count": len(feature_names),
            **shap_summary,
        }

        with open(resolved_output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

        return summary, 0
    except ExplainError as exc:
        return (
            {
                "status": exc.status,
                "error": str(exc),
                "artifact_dir": artifact_dir,
                "dataset": dataset,
            },
            1,
        )
    except Exception as exc:
        return (
            {
                "status": STATUS_EXPLAIN_FAILED,
                "error": str(exc),
                "artifact_dir": artifact_dir,
                "dataset": dataset,
            },
            1,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optional SHAP analysis for accepted artifacts")
    parser.add_argument(
        "--workspace",
        default=".",
        help=(
            "Workspace root containing the prepared data and accepted artifacts "
            "(default: current directory)."
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Path to an accepted artifact directory under the workspace models/accepted/...",
    )
    parser.add_argument(
        "--dataset",
        choices=["train", "test"],
        default=DEFAULT_DATASET,
        help="Prepared-data split to explain (default: test).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Maximum number of rows to explain (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top mean-absolute SHAP features to include (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--output-path",
        help="Optional override for the explanation summary JSON path.",
    )
    parser.add_argument(
        "--allow-prepared-data-mismatch",
        action="store_true",
        help="Proceed even if current prepared data does not match the artifact manifest.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    summary, exit_code = build_explanation_summary(
        artifact_dir=args.artifact_dir,
        dataset=args.dataset,
        sample_size=args.sample_size,
        top_k=args.top_k,
        output_path=args.output_path,
        allow_prepared_data_mismatch=args.allow_prepared_data_mismatch,
        workspace=args.workspace,
    )
    print_json(summary)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
