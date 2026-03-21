"""
run_experiment.py — Immutable experiment harness for corrugated price prediction.

This file owns:
- prepared-data loading
- experiment budget enforcement
- evaluation and logging
- git keep/discard semantics for autonomous mode

The only mutable research surface is `train.py`.
"""

import argparse
import ast
from datetime import datetime
import hashlib
import importlib.util
import json
import multiprocessing as mp
from queue import Empty
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import xgboost as xgb

RESULTS_PATH = "experiments/results.json"
RESULTS_CSV_PATH = "experiments/results.csv"
SESSION_STATE_PATH = "experiments/research_state.json"
TRAIN_PATH = "train.py"
PROGRAM_MD = "program.md"
META_PATH = "data/columns.json"
TRAIN_DATA_PATH = "data/train.parquet"
TEST_DATA_PATH = "data/test.parquet"
EXPERIMENT_BUDGET_S = 300
MODEL_PATH = "models/best_model.pkl"
FEATURES_PATH = "models/best_features.pkl"
BEST_TRAIN_SPEC_PATH = "models/best_train.py"
FROZEN_INPUT_PATHS = [
    "compile.py",
    "feature_spec.json",
    "prepare.py",
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    META_PATH,
]
STATUS_OK = "ok"
STATUS_BUDGET_EXCEEDED = "budget_exceeded"
STATUS_INVALID_CANDIDATE = "invalid_candidate"
STATUS_IMMUTABLE_VIOLATION = "immutable_violation"
STATUS_TRAIN_FAILED = "train_failed"
SUPPORTED_MODEL_FAMILIES = {
    "xgboost",
    "lightgbm",
    "ridge",
    "lasso",
    "elasticnet",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
}


class HarnessValidationError(Exception):
    status = STATUS_INVALID_CANDIDATE


class ImmutableViolationError(HarnessValidationError):
    status = STATUS_IMMUTABLE_VIOLATION


class InvalidCandidateError(HarnessValidationError):
    status = STATUS_INVALID_CANDIDATE


def load_results() -> list:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def git(args, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise RuntimeError(msg)
    return result


def in_git_repo() -> bool:
    return git(["rev-parse", "--is-inside-work-tree"], check=False).returncode == 0


def current_head() -> str:
    return git(["rev-parse", "--short", "HEAD"]).stdout.strip()


def current_branch() -> str:
    return git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def train_file_dirty() -> bool:
    return bool(git(["status", "--porcelain", "--", TRAIN_PATH]).stdout.strip())


def ensure_git_ready():
    if not in_git_repo():
        sys.exit(
            "[runner] ERROR: --auto requires a git repository.\n"
            "         Run: git init && git add . && git commit -m 'Initial snapshot'"
        )

    branch = current_branch()
    if train_file_dirty():
        print("[runner] Snapshotting current train.py into git before autonomous loop...")
        try:
            git(["add", "--", TRAIN_PATH])
            git(["commit", "-m", "autoresearch: baseline train.py snapshot", "--", TRAIN_PATH])
        except RuntimeError as exc:
            sys.exit(
                "[runner] ERROR: Failed to commit the current train.py baseline.\n"
                f"         {exc}"
            )

    print(f"[runner] Git branch: {branch}")
    print(f"[runner] Accepted train.py baseline: {current_head()}")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_frozen_inputs() -> dict:
    snapshot = {}
    missing = [path for path in FROZEN_INPUT_PATHS if not os.path.exists(path)]
    if missing:
        raise RuntimeError(
            "Missing prepared research inputs: "
            + ", ".join(missing)
            + ". Run compile.py / prepare.py first."
        )

    for path in FROZEN_INPUT_PATHS:
        snapshot[path] = sha256_file(path)
    return snapshot


def ensure_research_session():
    os.makedirs("experiments", exist_ok=True)
    current_snapshot = snapshot_frozen_inputs()

    if not os.path.exists(SESSION_STATE_PATH):
        state = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "budget_s": EXPERIMENT_BUDGET_S,
            "frozen_inputs": current_snapshot,
        }
        with open(SESSION_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
        print(f"[runner] Created research session baseline → {SESSION_STATE_PATH}")
        return state

    with open(SESSION_STATE_PATH) as f:
        state = json.load(f)

    baseline = state.get("frozen_inputs", {})
    mismatches = [path for path, digest in current_snapshot.items() if baseline.get(path) != digest]
    if mismatches:
        mismatch_str = ", ".join(mismatches)
        raise RuntimeError(
            "Frozen research inputs changed since the session baseline: "
            f"{mismatch_str}. Re-run setup, then delete {SESSION_STATE_PATH} to start a new "
            "research session."
        )

    return state


def load_prepared_data():
    for path in [TRAIN_DATA_PATH, TEST_DATA_PATH, META_PATH]:
        if not os.path.exists(path):
            raise RuntimeError(f"{path} not found. Run prepare.py first.")

    train = pd.read_parquet(TRAIN_DATA_PATH)
    test = pd.read_parquet(TEST_DATA_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return train, test, meta


def _is_docstring_expr(node) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _validate_import(node, module_name: str, alias_name: str):
    if not isinstance(node, ast.Import) or len(node.names) != 1:
        raise ImmutableViolationError(
            f"Immutable train.py header changed. Expected `import {module_name} as {alias_name}`."
        )
    alias = node.names[0]
    if alias.name != module_name or alias.asname != alias_name:
        raise ImmutableViolationError(
            f"Immutable train.py header changed. Expected `import {module_name} as {alias_name}`."
        )


def _validate_description_assignment(node):
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        raise ImmutableViolationError(
            "Immutable train.py structure changed around EXPERIMENT_DESCRIPTION."
        )
    target = node.targets[0]
    if not isinstance(target, ast.Name) or target.id != "EXPERIMENT_DESCRIPTION":
        raise ImmutableViolationError(
            "Immutable train.py structure changed around EXPERIMENT_DESCRIPTION."
        )


def _validate_function_signature(node, name: str, arg_names: list):
    if not isinstance(node, ast.FunctionDef) or node.name != name:
        raise ImmutableViolationError(f"Expected immutable train.py structure for `{name}`.")
    if node.decorator_list:
        raise ImmutableViolationError(f"`{name}` may not use decorators.")
    actual_args = [arg.arg for arg in node.args.args]
    if actual_args != arg_names:
        raise ImmutableViolationError(
            f"`{name}` must keep the immutable signature `{name}({', '.join(arg_names)})`."
        )


def _validate_main_guard(node):
    if not isinstance(node, ast.If):
        raise ImmutableViolationError("train.py must keep the immutable __main__ entrypoint.")
    test = node.test
    if not (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    ):
        raise ImmutableViolationError("train.py must keep the immutable __main__ entrypoint.")

    if len(node.body) != 2:
        raise ImmutableViolationError("train.py __main__ entrypoint may not be modified.")

    import_stmt, call_stmt = node.body
    if not (
        isinstance(import_stmt, ast.ImportFrom)
        and import_stmt.module == "run_experiment"
        and len(import_stmt.names) == 1
        and import_stmt.names[0].name == "run_single_from_train_entrypoint"
    ):
        raise ImmutableViolationError("train.py __main__ entrypoint may not be modified.")

    if not (
        isinstance(call_stmt, ast.Expr)
        and isinstance(call_stmt.value, ast.Call)
        and isinstance(call_stmt.value.func, ast.Name)
        and call_stmt.value.func.id == "run_single_from_train_entrypoint"
        and not call_stmt.value.args
        and not call_stmt.value.keywords
    ):
        raise ImmutableViolationError("train.py __main__ entrypoint may not be modified.")


def validate_train_source(train_code: str):
    try:
        tree = ast.parse(train_code, filename=TRAIN_PATH)
    except SyntaxError as exc:
        raise InvalidCandidateError(f"train.py is not valid Python: {exc}") from exc

    body = list(tree.body)
    if body and _is_docstring_expr(body[0]):
        body = body[1:]

    if len(body) != 6:
        raise ImmutableViolationError(
            "Only the immutable header plus EXPERIMENT_DESCRIPTION, engineer_features, "
            "and get_model_config are allowed in train.py."
        )

    _validate_import(body[0], "numpy", "np")
    _validate_import(body[1], "pandas", "pd")
    _validate_description_assignment(body[2])
    _validate_function_signature(body[3], "engineer_features", ["df", "meta"])
    _validate_function_signature(body[4], "get_model_config", [])
    _validate_main_guard(body[5])


def extract_description_from_source(train_code: str) -> str:
    try:
        tree = ast.parse(train_code, filename=TRAIN_PATH)
    except SyntaxError:
        match = re.search(r'EXPERIMENT_DESCRIPTION\s*=\s*\(\s*"([^"]+)"', train_code, re.DOTALL)
        return match.group(1).strip() if match else "unparseable train.py"

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "EXPERIMENT_DESCRIPTION":
                try:
                    value = ast.literal_eval(node.value)
                except Exception:
                    return "experiment description unavailable"
                if isinstance(value, str):
                    return " ".join(value.split())
    return "experiment description unavailable"


def load_train_spec():
    with open(TRAIN_PATH) as f:
        train_code = f.read()

    validate_train_source(train_code)

    module_name = f"train_runtime_{int(time.time() * 1000000)}"
    spec = importlib.util.spec_from_file_location(module_name, TRAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Failed to load train.py module.")
    spec.loader.exec_module(module)
    return module, train_code


def normalize_model_config(raw_config: dict) -> dict:
    if not isinstance(raw_config, dict):
        raise InvalidCandidateError("get_model_config() must return a dict.")

    family = raw_config.get("family")
    params = raw_config.get("params", {})

    if not isinstance(family, str):
        raise InvalidCandidateError("Model config must include a string `family`.")
    if family not in SUPPORTED_MODEL_FAMILIES:
        raise InvalidCandidateError(
            f"Unsupported model family `{family}`. Supported families: "
            + ", ".join(sorted(SUPPORTED_MODEL_FAMILIES))
        )
    if not isinstance(params, dict):
        raise InvalidCandidateError("Model config `params` must be a dict.")

    return {"family": family, "params": params}


def build_model(model_config: dict):
    family = model_config["family"]
    params = dict(model_config.get("params", {}))

    if family == "xgboost":
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "random_state": 42,
            "n_jobs": 1,
            "verbosity": 0,
        }
        defaults.update(params)
        defaults["random_state"] = 42
        defaults["n_jobs"] = 1
        defaults["verbosity"] = 0
        return xgb.XGBRegressor(**defaults)

    if family == "lightgbm":
        try:
            import lightgbm as lgb
        except OSError as exc:
            raise RuntimeError(
                "lightgbm requires libomp on this machine. Install libomp or use another "
                f"model family. Original error: {exc}"
            ) from exc
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": -1,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": 1,
            "verbose": -1,
        }
        defaults.update(params)
        defaults["random_state"] = 42
        defaults["n_jobs"] = 1
        defaults["verbose"] = -1
        return lgb.LGBMRegressor(**defaults)

    if family == "ridge":
        defaults = {"alpha": 1.0}
        defaults.update(params)
        return Ridge(**defaults)

    if family == "lasso":
        defaults = {"alpha": 0.001, "max_iter": 20000}
        defaults.update(params)
        return Lasso(**defaults)

    if family == "elasticnet":
        defaults = {"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 20000}
        defaults.update(params)
        return ElasticNet(**defaults)

    if family == "random_forest":
        defaults = {"n_estimators": 300, "random_state": 42, "n_jobs": 1}
        defaults.update(params)
        defaults["random_state"] = 42
        defaults["n_jobs"] = 1
        return RandomForestRegressor(**defaults)

    if family == "extra_trees":
        defaults = {"n_estimators": 300, "random_state": 42, "n_jobs": 1}
        defaults.update(params)
        defaults["random_state"] = 42
        defaults["n_jobs"] = 1
        return ExtraTreesRegressor(**defaults)

    defaults = {"random_state": 42}
    defaults.update(params)
    defaults["random_state"] = 42
    return GradientBoostingRegressor(**defaults)


def remaining_budget(start_time: float) -> float:
    return EXPERIMENT_BUDGET_S - (time.monotonic() - start_time)


def make_result_entry(
    *,
    status: str,
    description: str,
    elapsed_s: float,
    budget_s: int = EXPERIMENT_BUDGET_S,
    mape=None,
    cv_mape=None,
    rmse=None,
    r2=None,
    n_features=0,
    model_class=None,
    features=None,
    error=None,
    artifact_paths=None,
):
    return {
        "status": status,
        "budget_s": budget_s,
        "mape": round(float(mape), 4) if mape is not None else None,
        "cv_mape": round(float(cv_mape), 4) if cv_mape is not None else None,
        "rmse": round(float(rmse), 4) if rmse is not None else None,
        "r2": round(float(r2), 4) if r2 is not None else None,
        "elapsed_s": round(float(elapsed_s), 1),
        "n_features": int(n_features),
        "model_class": model_class,
        "description": description,
        "is_best": False,
        "features": features or [],
        "error": error,
        "artifact_paths": artifact_paths or {},
    }


def best_completed_result(results: list):
    completed = [
        r for r in results
        if r.get("status", STATUS_OK) == STATUS_OK and r.get("mape") is not None
    ]
    if not completed:
        return None
    return min(completed, key=lambda r: r["mape"])


def best_completed_mape(results: list):
    best = best_completed_result(results)
    return best["mape"] if best else float("inf")


def save_result(entry: dict):
    results = load_results()
    clean_entry = dict(entry)
    clean_entry["timestamp"] = datetime.now().isoformat(timespec="seconds")
    clean_entry["exp_id"] = len(results)
    clean_entry.pop("artifact_paths", None)
    results.append(clean_entry)

    os.makedirs("experiments", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(results).to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\n  Logged as experiment #{clean_entry['exp_id']} → {RESULTS_PATH}")
    return clean_entry


def persist_best_artifacts(entry: dict, train_source: str):
    artifact_paths = entry.get("artifact_paths", {})
    if not artifact_paths:
        return

    os.makedirs("models", exist_ok=True)
    shutil.copyfile(artifact_paths["model"], MODEL_PATH)
    shutil.copyfile(artifact_paths["features"], FEATURES_PATH)
    with open(BEST_TRAIN_SPEC_PATH, "w") as f:
        f.write(train_source)


def cleanup_artifacts(entry: dict):
    for path in entry.get("artifact_paths", {}).values():
        if path and os.path.exists(path):
            os.remove(path)


def print_run_report(entry: dict):
    print(f"\n{'─' * 60}")
    print(f"  {entry['description']}")
    print(f"{'─' * 60}")
    print(f"  Status   : {entry['status']}")
    print(f"  Budget   : {entry.get('budget_s', EXPERIMENT_BUDGET_S)}s")
    print(f"  Elapsed  : {entry.get('elapsed_s', 0):.1f}s")

    if entry.get("model_class"):
        print(f"  Model    : {entry['model_class']}")

    if entry["status"] == STATUS_OK:
        print(f"  Features ({entry['n_features']}): {entry.get('features', [])}")
        print(f"  Test MAPE: {entry['mape']:.2f}%   ← PRIMARY METRIC")
        print(f"  Test RMSE: {entry['rmse']:.4f}")
        print(f"  Test R²  : {entry['r2']:.4f}")
        if entry.get("is_best"):
            print("  ✅ NEW BEST — saved to models/best_model.pkl")
        else:
            print("  ❌ Not best")
    else:
        if entry.get("error"):
            print(f"  Error    : {entry['error']}")
    print(f"{'─' * 60}")


def print_summary():
    results = load_results()
    if not results:
        print("No experiments run yet. Run: python run_experiment.py")
        return

    df = pd.DataFrame(results)
    if "status" not in df.columns:
        df["status"] = STATUS_OK
    if "budget_s" not in df.columns:
        df["budget_s"] = EXPERIMENT_BUDGET_S
    if "mape" not in df.columns:
        df["mape"] = np.nan

    df["sort_status"] = np.where(df["status"] == STATUS_OK, 0, 1)
    df["sort_mape"] = df["mape"].fillna(np.inf)

    cols = [
        c for c in [
            "exp_id",
            "status",
            "budget_s",
            "mape",
            "rmse",
            "r2",
            "model_class",
            "n_features",
            "description",
        ]
        if c in df.columns
    ]

    print(f"\n{'═' * 80}")
    print("  EXPERIMENT HISTORY — successful runs first, then failures/timeouts")
    print(f"{'═' * 80}")
    print(
        df.sort_values(["sort_status", "sort_mape", "exp_id"])[cols].to_string(index=False)
    )

    best = best_completed_result(results)
    if best is None:
        print("\n  No successful completed experiments yet.")
    else:
        print(
            f"\n  🏆 Best: Exp #{int(best['exp_id'])} | "
            f"MAPE={best['mape']:.2f}% | {best['description']}"
        )


def show_best():
    results = load_results()
    best = best_completed_result(results)
    if best is None:
        print("No successful experiments yet.")
        return

    print(f"\n{'═' * 60}")
    print(f"  🏆 BEST MODEL — Experiment #{best['exp_id']}")
    print(f"{'═' * 60}")
    for key, value in best.items():
        if key != "features":
            print(f"  {key:20s}: {value}")
    print(f"\n  Features: {best.get('features', 'N/A')}")
    print("\n  Load model:")
    print("    import joblib")
    print("    model    = joblib.load('models/best_model.pkl')")
    print("    features = joblib.load('models/best_features.pkl')")
    print("    best_train_spec = 'models/best_train.py'")


def _fit_and_score_worker(
    queue,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    model_config,
    artifact_dir,
):
    try:
        model = build_model(model_config)
        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), 0, None)

        model_path = os.path.join(artifact_dir, "model.pkl")
        features_path = os.path.join(artifact_dir, "features.pkl")
        joblib.dump(model, model_path)
        joblib.dump(feature_names, features_path)

        queue.put(
            make_result_entry(
                status=STATUS_OK,
                description="",
                elapsed_s=0.0,
                mape=mean_absolute_percentage_error(y_test, y_pred) * 100,
                cv_mape=None,
                rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
                r2=r2_score(y_test, y_pred),
                n_features=len(feature_names),
                model_class=type(model).__name__,
                features=feature_names,
                artifact_paths={"model": model_path, "features": features_path},
            )
        )
    except Exception as exc:
        queue.put(
            make_result_entry(
                status=STATUS_TRAIN_FAILED,
                description="",
                elapsed_s=0.0,
                model_class=model_config.get("family"),
                error=str(exc),
                features=feature_names,
                n_features=len(feature_names),
            )
        )


def fit_and_score_with_budget(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names: list,
    model_config: dict,
    start_time: float,
):
    budget_left = remaining_budget(start_time)
    if budget_left <= 0:
        return make_result_entry(
            status=STATUS_BUDGET_EXCEEDED,
            description="",
            elapsed_s=EXPERIMENT_BUDGET_S,
            model_class=model_config["family"],
            error="Budget exhausted before model fitting started.",
            features=feature_names,
            n_features=len(feature_names),
        )

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    with tempfile.TemporaryDirectory() as tmpdir:
        process = ctx.Process(
            target=_fit_and_score_worker,
            args=(queue, X_train, y_train, X_test, y_test, feature_names, model_config, tmpdir),
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
            return make_result_entry(
                status=STATUS_BUDGET_EXCEEDED,
                description="",
                elapsed_s=EXPERIMENT_BUDGET_S,
                model_class=model_config["family"],
                error=f"Experiment exceeded the fixed {EXPERIMENT_BUDGET_S}s budget.",
                features=feature_names,
                n_features=len(feature_names),
            )

        try:
            result = queue.get(timeout=1)
        except Empty:
            queue.close()
            queue.join_thread()
            return make_result_entry(
                status=STATUS_TRAIN_FAILED,
                description="",
                elapsed_s=EXPERIMENT_BUDGET_S - max(remaining_budget(start_time), 0),
                model_class=model_config["family"],
                error="Training process exited without returning results.",
                features=feature_names,
                n_features=len(feature_names),
            )
        queue.close()
        queue.join_thread()

        artifact_paths = result.get("artifact_paths", {})
        copied_artifacts = {}
        for key, path in artifact_paths.items():
            if path:
                fd, copied_path = tempfile.mkstemp(prefix=f"corrugated_{key}_", suffix=".pkl")
                os.close(fd)
                shutil.copyfile(path, copied_path)
                copied_artifacts[key] = copied_path
        result["artifact_paths"] = copied_artifacts
        return result


def run_single_experiment():
    ensure_research_session()

    with open(TRAIN_PATH) as f:
        train_source = f.read()

    description = extract_description_from_source(train_source)

    try:
        train_spec, train_source = load_train_spec()
    except HarnessValidationError as exc:
        entry = make_result_entry(
            status=exc.status,
            description=description,
            elapsed_s=0.0,
            error=str(exc),
        )
        print_run_report(entry)
        return save_result(entry)

    train_df, test_df, meta = load_prepared_data()
    start_time = time.monotonic()

    try:
        X_train_raw = train_df.drop(columns=["price_msf"])
        y_train = train_df["price_msf"].values
        X_test_raw = test_df.drop(columns=["price_msf"])
        y_test = test_df["price_msf"].values

        X_train = train_spec.engineer_features(X_train_raw, meta)
        X_test = train_spec.engineer_features(X_test_raw, meta)

        X_train = X_train.select_dtypes(include=[np.number])
        X_test = X_test.select_dtypes(include=[np.number])
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        feature_names = list(X_train.columns)
        if not feature_names:
            raise InvalidCandidateError("engineer_features() returned no numeric features.")

        model_config = normalize_model_config(train_spec.get_model_config())
    except HarnessValidationError as exc:
        entry = make_result_entry(
            status=exc.status,
            description=description,
            elapsed_s=EXPERIMENT_BUDGET_S - max(remaining_budget(start_time), 0),
            error=str(exc),
        )
        print_run_report(entry)
        return save_result(entry)
    except Exception as exc:
        entry = make_result_entry(
            status=STATUS_TRAIN_FAILED,
            description=description,
            elapsed_s=EXPERIMENT_BUDGET_S - max(remaining_budget(start_time), 0),
            error=str(exc),
        )
        print_run_report(entry)
        return save_result(entry)

    worker_result = fit_and_score_with_budget(
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names,
        model_config,
        start_time,
    )
    worker_result["description"] = description
    worker_result["elapsed_s"] = EXPERIMENT_BUDGET_S - max(remaining_budget(start_time), 0)

    previous_results = load_results()
    previous_best = best_completed_mape(previous_results)
    if worker_result["status"] == STATUS_OK and worker_result["mape"] < previous_best:
        worker_result["is_best"] = True
        persist_best_artifacts(worker_result, train_source)

    print_run_report(worker_result)
    saved = save_result(worker_result)
    cleanup_artifacts(worker_result)
    return saved


def run_single_from_train_entrypoint():
    try:
        run_single_experiment()
    except RuntimeError as exc:
        sys.exit(f"[runner] ERROR: {exc}")
    print_summary()


def build_candidate_commit_message(results: list, train_code: str) -> str:
    exp_num = len(results)
    desc = extract_description_from_source(train_code)
    desc = " ".join(desc.split())
    if len(desc) > 60:
        desc = desc[:57] + "..."
    return f"autoresearch: exp {exp_num} {desc}"


def commit_candidate(train_code: str, results: list) -> Optional[str]:
    with open(TRAIN_PATH, "w") as f:
        f.write(train_code)

    if not train_file_dirty():
        return None

    git(["add", "--", TRAIN_PATH])
    git(["commit", "-m", build_candidate_commit_message(results, train_code), "--", TRAIN_PATH])
    return current_head()


def discard_candidate_commit():
    git(["reset", "--soft", "HEAD~1"])
    git(["restore", "--source=HEAD", "--staged", "--worktree", "--", TRAIN_PATH])


def get_agent_suggestion(results: list, train_code: str, program_md: str, meta: dict) -> str:
    try:
        import anthropic
    except ImportError:
        sys.exit("[runner] anthropic package not installed. Run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit(
            "[runner] ANTHROPIC_API_KEY not set.\n"
            "         export ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = anthropic.Anthropic(api_key=api_key)
    history_str = json.dumps(results[-10:], indent=2)
    meta_str = json.dumps(meta, indent=2)

    prompt = f"""You are an ML research agent improving a price prediction model.

Your instructions are in program.md:
<program_md>
{program_md}
</program_md>

Frozen prepared-data metadata:
<columns_json>
{meta_str}
</columns_json>

Current mutable train.py:
<train_py>
{train_code}
</train_py>

Experiment history (last 10 runs):
<history>
{history_str}
</history>

Return the COMPLETE updated train.py.

Hard constraints:
- You may only change EXPERIMENT_DESCRIPTION, engineer_features(), and get_model_config()
- Do not add imports, helpers, globals, or any new top-level code
- get_model_config() must return a dict with:
  - family: one of {sorted(SUPPORTED_MODEL_FAMILIES)}
  - params: dict
- Optimize for lower test MAPE under a fixed {EXPERIMENT_BUDGET_S}s experiment budget
- Return ONLY raw Python source, with no markdown fences or extra prose
"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def autonomous_loop(n: int):
    print(f"\n[runner] Starting autonomous loop: {n} experiments")
    ensure_git_ready()

    try:
        ensure_research_session()
    except RuntimeError as exc:
        print(f"[runner] ERROR: {exc}")
        return
    with open(PROGRAM_MD) as f:
        program_md = f.read()
    with open(META_PATH) as f:
        meta = json.load(f)

    pending_candidate = None

    for i in range(n):
        print(f"\n{'═' * 60}")
        print(f"  LOOP {i + 1}/{n}")
        print(f"{'═' * 60}")

        try:
            entry = run_single_experiment()
        except RuntimeError as exc:
            print(f"[runner] ERROR: {exc}")
            break

        if pending_candidate is not None:
            improved = (
                entry.get("status") == STATUS_OK
                and entry.get("mape") is not None
                and entry["mape"] < pending_candidate["best_mape_before"]
            )

            if improved:
                print(
                    "[runner] Candidate improved MAPE "
                    f"({entry['mape']:.2f}% < {pending_candidate['best_mape_before']:.2f}%)"
                )
                print(f"[runner] Keeping git commit {pending_candidate['commit']}")
            else:
                reason = entry.get("status", STATUS_TRAIN_FAILED)
                print(f"[runner] Candidate rejected with status `{reason}`")
                try:
                    discard_candidate_commit()
                    print(f"[runner] Discarded git commit {pending_candidate['commit']}")
                    print(f"[runner] Reset train.py to accepted commit {current_head()}")
                except RuntimeError as exc:
                    print(f"[runner] ERROR: failed to discard candidate cleanly: {exc}")
                    break
            pending_candidate = None

        if i == n - 1:
            print("[runner] Final experiment complete.")
            break

        print("[runner] Asking agent for next hypothesis...")
        with open(TRAIN_PATH) as f:
            train_code = f.read()

        results = load_results()
        new_train = get_agent_suggestion(results, train_code, program_md, meta)

        if not new_train.strip():
            rejected = make_result_entry(
                status=STATUS_INVALID_CANDIDATE,
                description="empty agent response",
                elapsed_s=0.0,
                error="Agent returned an empty train.py candidate.",
            )
            print_run_report(rejected)
            save_result(rejected)
            break

        if new_train.startswith("```"):
            lines = new_train.splitlines()
            new_train = "\n".join(lines[1:-1])

        try:
            validate_train_source(new_train)
        except HarnessValidationError as exc:
            rejected = make_result_entry(
                status=exc.status,
                description=extract_description_from_source(new_train),
                elapsed_s=0.0,
                error=str(exc),
            )
            print_run_report(rejected)
            save_result(rejected)
            print("[runner] Agent suggestion rejected before commit.")
            break

        best_mape_before = best_completed_mape(results)

        try:
            commit_hash = commit_candidate(new_train, results)
        except RuntimeError as exc:
            print(f"[runner] ERROR: failed to create candidate git commit: {exc}")
            break

        if commit_hash is None:
            rejected = make_result_entry(
                status=STATUS_INVALID_CANDIDATE,
                description=extract_description_from_source(new_train),
                elapsed_s=0.0,
                error="Agent candidate did not change train.py.",
            )
            print_run_report(rejected)
            save_result(rejected)
            print("[runner] Agent produced no train.py change — stopping loop")
            break

        pending_candidate = {
            "commit": commit_hash,
            "best_mape_before": best_mape_before,
        }
        print(f"[runner] Candidate committed as {commit_hash}")

        time.sleep(1)

    print_summary()


def manual_loop(n: int):
    print(f"\n[runner] Manual loop: running the immutable harness {n} time(s)")
    for i in range(n):
        print(f"\n{'─' * 40}  Run {i + 1}/{n}  {'─' * 40}")
        try:
            run_single_experiment()
        except RuntimeError as exc:
            print(f"[runner] ERROR: {exc}")
            break
        time.sleep(0.2)
    print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Immutable AutoResearch experiment harness")
    parser.add_argument("--n", type=int, default=1, help="Number of experiments to run")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Enable git-backed autonomous mode (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument("--summary", action="store_true", help="Print experiment history and exit")
    parser.add_argument("--best", action="store_true", help="Show best model details and exit")
    args = parser.parse_args()

    os.makedirs("experiments", exist_ok=True)

    if args.summary:
        print_summary()
    elif args.best:
        show_best()
    elif args.auto:
        autonomous_loop(args.n)
    else:
        manual_loop(args.n)
