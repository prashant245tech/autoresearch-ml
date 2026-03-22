"""
prepare.py — Generic data preparation pipeline.
Reads config/feature_spec.json and executes it deterministically. No LLM calls here.
See config/feature_spec.schema.json for the prep-spec contract.

Run order:
    agent writes config/feature_spec.json                               (once)
    python prepare.py    ← executes spec → train/test parquet          (once per data change)
    python train.py      ← agent iterates on this
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from workspace_paths import WorkspacePaths

warnings.filterwarnings("ignore")

DEFAULT_SPEC_LABEL = "config/feature_spec.json"

ALLOWED_FEATURE_TYPES = {"numeric", "ordinal", "categorical", "boolean"}
ALLOWED_FILL_STRATEGIES = {"value", "median", "mean", "mode", "none"}
ALLOWED_NORMALISE = {"none", "uppercase_strip"}
ALLOWED_FILTER_OPS = {
    "eq",
    "ne",
    "in",
    "not_in",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
    "is_null",
    "not_null",
}
ALLOWED_TOP_LEVEL_KEYS = {
    "$schema",
    "version",
    "data_file",
    "target_column",
    "test_data_file",
    "test_target_column",
    "test_size",
    "random_state",
    "outlier_iqr_k",
    "domain_context",
    "row_filters",
    "train_row_filters",
    "test_row_filters",
    "filter_specs",
    "features",
}
ALLOWED_ROW_FILTER_KEYS = {"column", "op", "value"}
ALLOWED_FILL_NULL_KEYS = {"strategy", "value"}
ALLOWED_CLIP_KEYS = {"min", "max"}
ALLOWED_FEATURE_SPEC_KEYS = {
    "column",
    "type",
    "required",
    "drop_if_null",
    "fill_null",
    "clip",
    "normalise",
    "ordinal_order",
    "categorical_consolidation",
    "unknown_sentinel",
    "notes",
}
ALLOWED_FILTER_SPEC_KEYS = {
    "column",
    "type",
    "fill_null",
    "clip",
    "normalise",
    "ordinal_order",
    "categorical_consolidation",
    "unknown_sentinel",
    "notes",
}


def _fail(message: str) -> None:
    sys.exit(f"[prepare] ERROR: {message}")


def _reject_unknown_keys(payload: dict, allowed_keys: set[str], label: str) -> None:
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        _fail(f"{label} has unsupported keys: {', '.join(unknown_keys)}.")


def _validate_column_spec(item: dict, label: str, allowed_keys: set[str]) -> dict:
    if not isinstance(item, dict):
        _fail(f"{label} must be an object.")
    _reject_unknown_keys(item, allowed_keys, label)

    if not isinstance(item.get("column"), str) or not item["column"].strip():
        _fail(f"{label}.column must be a non-empty string.")

    ftype = item.get("type")
    if ftype not in ALLOWED_FEATURE_TYPES:
        _fail(f"{label}.type must be one of {sorted(ALLOWED_FEATURE_TYPES)}.")

    fill_cfg = item.get("fill_null", {"strategy": "none"})
    if not isinstance(fill_cfg, dict):
        _fail(f"{label}.fill_null must be an object.")
    _reject_unknown_keys(fill_cfg, ALLOWED_FILL_NULL_KEYS, f"{label}.fill_null")
    strategy = fill_cfg.get("strategy", "none")
    if strategy not in ALLOWED_FILL_STRATEGIES:
        _fail(
            f"{label}.fill_null.strategy must be one of "
            f"{sorted(ALLOWED_FILL_STRATEGIES)}."
        )
    if strategy == "value" and "value" not in fill_cfg:
        _fail(f"{label}.fill_null.value is required.")

    normalise = item.get("normalise", "none")
    if normalise not in ALLOWED_NORMALISE:
        _fail(f"{label}.normalise must be one of {sorted(ALLOWED_NORMALISE)}.")

    clip_cfg = item.get("clip")
    if clip_cfg is not None:
        if not isinstance(clip_cfg, dict):
            _fail(f"{label}.clip must be an object.")
        _reject_unknown_keys(clip_cfg, ALLOWED_CLIP_KEYS, f"{label}.clip")

    if ftype == "ordinal":
        order = item.get("ordinal_order")
        if not isinstance(order, list) or not order:
            _fail(f"{label}.ordinal_order must be a non-empty list.")

    if ftype == "categorical":
        consolidation = item.get("categorical_consolidation", {})
        if consolidation is not None and not isinstance(consolidation, dict):
            _fail(f"{label}.categorical_consolidation must be an object.")

    return item


def validate_column_specs(
    specs,
    label: str,
    require_non_empty: bool = False,
    spec_label: str = DEFAULT_SPEC_LABEL,
) -> list:
    if specs is None:
        specs = []
    if not isinstance(specs, list):
        _fail(f"{label} must be a list.")
    if require_non_empty and not specs:
        _fail(f"{spec_label} `{label}` must be a non-empty list.")

    validated = []
    seen_columns = set()
    allowed_keys = ALLOWED_FEATURE_SPEC_KEYS if label == "features" else ALLOWED_FILTER_SPEC_KEYS
    for idx, item in enumerate(specs):
        item_label = f"{label}[{idx}]"
        validated_item = _validate_column_spec(item, item_label, allowed_keys)
        column = validated_item["column"]
        if column in seen_columns:
            _fail(f"{label} contains duplicate column definitions for '{column}'.")
        seen_columns.add(column)
        validated.append(validated_item)

    return validated


def validate_row_filters(filters, label: str) -> list:
    if filters is None:
        return []
    if not isinstance(filters, list):
        _fail(f"{label} must be a list.")

    validated = []
    for idx, filt in enumerate(filters):
        filter_label = f"{label}[{idx}]"
        if not isinstance(filt, dict):
            _fail(f"{filter_label} must be an object.")
        _reject_unknown_keys(filt, ALLOWED_ROW_FILTER_KEYS, filter_label)

        column = filt.get("column")
        if not isinstance(column, str) or not column.strip():
            _fail(f"{filter_label}.column must be a non-empty string.")

        op = filt.get("op")
        if op not in ALLOWED_FILTER_OPS:
            _fail(f"{filter_label}.op must be one of {sorted(ALLOWED_FILTER_OPS)}.")

        needs_value = op not in {"is_null", "not_null"}
        if needs_value and "value" not in filt:
            _fail(f"{filter_label}.value is required for op={op}.")
        if not needs_value and "value" in filt:
            _fail(f"{filter_label}.value must not be set for op={op}.")

        if op in {"in", "not_in"} and not isinstance(filt.get("value"), list):
            _fail(f"{filter_label}.value must be a list for op={op}.")

        validated.append(filt)

    return validated


def validate_spec(spec: dict, spec_label: str = DEFAULT_SPEC_LABEL) -> dict:
    if not isinstance(spec, dict):
        _fail(f"{spec_label} must contain a top-level JSON object.")
    _reject_unknown_keys(spec, ALLOWED_TOP_LEVEL_KEYS, spec_label)

    required_top_level = ["data_file", "target_column", "features"]
    missing = [key for key in required_top_level if key not in spec]
    if missing:
        _fail(f"{spec_label} is missing required keys: " + ", ".join(missing))

    if not isinstance(spec["data_file"], str) or not spec["data_file"].strip():
        _fail(f"{spec_label} `data_file` must be a non-empty string.")
    if not isinstance(spec["target_column"], str) or not spec["target_column"].strip():
        _fail(f"{spec_label} `target_column` must be a non-empty string.")

    if "test_data_file" in spec and (
        not isinstance(spec["test_data_file"], str) or not spec["test_data_file"].strip()
    ):
        _fail(f"{spec_label} `test_data_file` must be a non-empty string when set.")
    if "test_target_column" in spec and (
        not isinstance(spec["test_target_column"], str)
        or not spec["test_target_column"].strip()
    ):
        _fail(f"{spec_label} `test_target_column` must be a non-empty string when set.")

    if "row_filters" in spec and "train_row_filters" in spec:
        _fail(f"Use either `row_filters` or `train_row_filters` in {spec_label}, not both.")

    spec["train_row_filters"] = validate_row_filters(
        spec.get("train_row_filters", spec.get("row_filters")),
        "train_row_filters",
    )
    spec["test_row_filters"] = validate_row_filters(
        spec.get("test_row_filters"),
        "test_row_filters",
    )
    spec["features"] = validate_column_specs(
        spec["features"],
        "features",
        require_non_empty=True,
        spec_label=spec_label,
    )
    spec["filter_specs"] = validate_column_specs(
        spec.get("filter_specs", []),
        "filter_specs",
        spec_label=spec_label,
    )

    if spec.get("test_target_column") and not spec.get("test_data_file"):
        _fail(f"{spec_label} `test_target_column` requires `test_data_file`.")
    if spec["test_row_filters"] and not spec.get("test_data_file"):
        _fail(f"{spec_label} `test_row_filters` requires `test_data_file`.")

    feature_columns = {feat["column"] for feat in spec["features"]}
    filter_spec_columns = {feat["column"] for feat in spec["filter_specs"]}
    overlapping_columns = sorted(feature_columns & filter_spec_columns)
    if overlapping_columns:
        _fail(
            "Columns must not appear in both `features` and `filter_specs`: "
            + ", ".join(overlapping_columns)
        )

    filtered_columns = {
        filt["column"] for filt in spec["train_row_filters"] + spec["test_row_filters"]
    }
    uncovered_filter_columns = sorted(filtered_columns - feature_columns - filter_spec_columns)
    if uncovered_filter_columns:
        _fail(
            "Each filtered column must be declared in `features` or `filter_specs`: "
            + ", ".join(uncovered_filter_columns)
        )

    return spec


def load_spec(paths: WorkspacePaths) -> dict:
    spec_path = paths.feature_spec_path
    schema_path = paths.feature_spec_schema_path
    spec_label = paths.display_path(spec_path)
    if not os.path.exists(spec_path):
        _fail(
            f"{spec_label} not found.\n"
            "          Create it directly from program.md and validate against "
            f"{paths.display_path(schema_path)}"
        )
    with open(spec_path) as handle:
        spec = json.load(handle)
    return validate_spec(spec, spec_label=spec_label)


def load_table(
    file_path: str,
    spec_label: str = DEFAULT_SPEC_LABEL,
    paths: WorkspacePaths | None = None,
) -> pd.DataFrame:
    workspace_paths = paths or WorkspacePaths.from_value(".")
    workspace_data_dir = workspace_paths.data_dir
    candidates = [
        file_path,
        f"/mnt/user-data/uploads/{os.path.basename(file_path)}",
        str(workspace_data_dir / os.path.basename(file_path)),
        f"data/{os.path.basename(file_path)}",
    ]
    for path in candidates:
        if os.path.exists(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
            print(f"[prepare] Loaded: {path}  →  {df.shape[0]:,} rows × {df.shape[1]} cols")
            return df
    _fail(
        f"File not found at '{file_path}'.\n"
        f"          Update program.md / {spec_label} and regenerate the prep spec."
    )


def apply_feature_spec(df: pd.DataFrame, feat: dict) -> pd.DataFrame:
    col = feat["column"]
    if col not in df.columns:
        print(f"[prepare] WARNING: '{col}' not in input file — skipping")
        return df

    ftype = feat.get("type", "numeric")
    normalise = feat.get("normalise", "none")
    fill_cfg = feat.get("fill_null", {})
    clip_cfg = feat.get("clip", {})
    consolidation = feat.get("categorical_consolidation", {})

    if normalise == "uppercase_strip":
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
        )

    if consolidation:
        upper_map = {str(k).strip().upper(): str(v).strip().upper() for k, v in consolidation.items()}
        df[col] = df[col].map(
            lambda x: upper_map.get(str(x).strip().upper(), x) if pd.notna(x) else x
        )

    strategy = fill_cfg.get("strategy", "none")
    if strategy == "value":
        df[col] = df[col].fillna(fill_cfg.get("value", np.nan))
    elif strategy == "median":
        num = pd.to_numeric(df[col], errors="coerce")
        df[col] = num.fillna(num.median())
    elif strategy == "mean":
        num = pd.to_numeric(df[col], errors="coerce")
        df[col] = num.fillna(num.mean())
    elif strategy == "mode":
        mode_vals = df[col].mode()
        if not mode_vals.empty:
            df[col] = df[col].fillna(mode_vals.iloc[0])

    if ftype in ("numeric", "boolean"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if ftype == "boolean":
            df[col] = df[col].fillna(0).clip(0, 1).astype(int)

    if ftype == "numeric":
        lo, hi = clip_cfg.get("min"), clip_cfg.get("max")
        if lo is not None:
            df[col] = df[col].clip(lower=lo)
        if hi is not None:
            df[col] = df[col].clip(upper=hi)

    return df


def drop_required_null_rows(df: pd.DataFrame, features: list, dataset_label: str) -> pd.DataFrame:
    drop_cols = [f["column"] for f in features if f.get("drop_if_null") and f["column"] in df.columns]
    if not drop_cols:
        return df

    before = len(df)
    df = df.dropna(subset=drop_cols)
    print(
        f"[prepare] Dropped {before - len(df):,} rows from {dataset_label} with nulls in: {drop_cols}"
    )
    return df


def apply_feature_cleaning(df: pd.DataFrame, features: list, dataset_label: str) -> pd.DataFrame:
    for feat in features:
        df = apply_feature_spec(df, feat)
    print(f"[prepare] Applied feature cleaning to {dataset_label}")
    return df


def apply_filter_spec_cleaning(
    df: pd.DataFrame, filter_specs: list, dataset_label: str
) -> pd.DataFrame:
    if not filter_specs:
        return df
    for feat in filter_specs:
        df = apply_feature_spec(df, feat)
    print(f"[prepare] Applied filter-spec cleaning to {dataset_label}")
    return df


def _build_filter_mask(series: pd.Series, op: str, value):
    if op == "eq":
        return series == value
    if op == "ne":
        return series != value
    if op == "in":
        return series.isin(value)
    if op == "not_in":
        return ~series.isin(value)
    if op == "gt":
        return pd.to_numeric(series, errors="coerce") > value
    if op == "gte":
        return pd.to_numeric(series, errors="coerce") >= value
    if op == "lt":
        return pd.to_numeric(series, errors="coerce") < value
    if op == "lte":
        return pd.to_numeric(series, errors="coerce") <= value
    if op == "contains":
        return series.astype(str).str.contains(str(value), case=False, na=False)
    if op == "is_null":
        return series.isna()
    if op == "not_null":
        return series.notna()
    raise ValueError(f"Unsupported filter op: {op}")


def apply_row_filters(df: pd.DataFrame, filters: list, dataset_label: str) -> pd.DataFrame:
    if not filters:
        return df

    for filt in filters:
        column = filt["column"]
        if column not in df.columns:
            _fail(f"{dataset_label} filter column '{column}' is not present in the loaded dataset.")

        before = len(df)
        mask = _build_filter_mask(df[column], filt["op"], filt.get("value"))
        df = df[mask].copy()
        print(
            f"[prepare] Applied {dataset_label} filter {column} {filt['op']} "
            f"{filt.get('value', '')!r}: {before:,} → {len(df):,} rows"
        )

    return df.reset_index(drop=True)


def clean_target(
    df: pd.DataFrame,
    target_col: str,
    spec: dict,
    dataset_label: str,
    apply_outlier_filter: bool = True,
) -> pd.DataFrame:
    before = len(df)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].notna() & ~np.isinf(df[target_col]) & (df[target_col] > 0)]

    if apply_outlier_filter:
        k = float(spec.get("outlier_iqr_k", 4.0))
        if k:
            q1, q3 = df[target_col].quantile([0.25, 0.75])
            iqr = q3 - q1
            df = df[(df[target_col] >= q1 - k * iqr) & (df[target_col] <= q3 + k * iqr)]

    print(
        f"[prepare] Target cleaned for {dataset_label}: {before:,} → {len(df):,} rows "
        f"({before - len(df):,} removed)"
    )
    return df.reset_index(drop=True)


def fit_encoding_meta(df: pd.DataFrame, features: list) -> dict:
    encoding_meta = {}

    for feat in features:
        col = feat["column"]
        ftype = feat.get("type", "numeric")
        if col not in df.columns:
            continue

        if ftype == "ordinal":
            encoding_meta[col] = {
                "type": "ordinal",
                "order": [str(v).strip().upper() for v in feat.get("ordinal_order", [])],
                "encoded_col": f"{col}_encoded",
                "sentinel": feat.get("unknown_sentinel", -1),
            }

        elif ftype == "categorical":
            sentinel = str(feat.get("unknown_sentinel", "UNKNOWN"))
            raw_vals = (
                df[col]
                .astype(str)
                .str.strip()
                .str.upper()
                .replace({"NAN": sentinel, "NONE": sentinel, "": sentinel})
            )
            classes = sorted(set(raw_vals.tolist()) | {sentinel})
            encoding_meta[col] = {
                "type": "categorical",
                "encoded_col": f"{col}_enc",
                "classes": classes,
                "sentinel": sentinel,
            }

    return encoding_meta


def apply_encodings(df: pd.DataFrame, features: list, encoding_meta: dict) -> pd.DataFrame:
    df = df.copy()

    for feat in features:
        col = feat["column"]
        ftype = feat.get("type", "numeric")
        if col not in df.columns:
            continue

        if ftype == "ordinal":
            meta = encoding_meta[col]
            mapping = {value: idx + 1 for idx, value in enumerate(meta["order"])}
            raw_vals = (
                df[col]
                .astype(str)
                .str.strip()
                .str.upper()
                .replace({"NAN": np.nan, "NONE": np.nan})
            )
            df[f"{col}_raw"] = raw_vals.fillna("UNKNOWN")
            df[meta["encoded_col"]] = raw_vals.map(mapping).fillna(meta["sentinel"]).astype(int)

        elif ftype == "categorical":
            meta = encoding_meta[col]
            sentinel = meta["sentinel"]
            raw_vals = (
                df[col]
                .astype(str)
                .str.strip()
                .str.upper()
                .replace({"NAN": sentinel, "NONE": sentinel, "": sentinel})
            )
            raw_vals = raw_vals.where(raw_vals.isin(meta["classes"]), sentinel)
            class_to_int = {value: idx for idx, value in enumerate(meta["classes"])}
            df[f"{col}_raw"] = raw_vals
            df[meta["encoded_col"]] = raw_vals.map(class_to_int).astype(int)

    return df


def build_matrix(df: pd.DataFrame, features: list, encoding_meta: dict, target_col: str):
    model_cols, ref_cols = [], []
    for feat in features:
        col = feat["column"]
        ftype = feat.get("type", "numeric")
        if col not in df.columns:
            continue

        if ftype in ("numeric", "boolean"):
            model_cols.append(col)
        elif ftype == "ordinal":
            encoded_col = encoding_meta.get(col, {}).get("encoded_col", f"{col}_encoded")
            if encoded_col in df.columns:
                model_cols.append(encoded_col)
            raw_col = f"{col}_raw"
            if raw_col in df.columns:
                ref_cols.append(raw_col)
        elif ftype == "categorical":
            encoded_col = encoding_meta.get(col, {}).get("encoded_col", f"{col}_enc")
            if encoded_col in df.columns:
                model_cols.append(encoded_col)
            raw_col = f"{col}_raw"
            if raw_col in df.columns:
                ref_cols.append(raw_col)

    all_cols = [col for col in model_cols + ref_cols + [target_col] if col in df.columns]
    return df[all_cols].reset_index(drop=True), model_cols, ref_cols


def prepare_dataset(
    raw_df: pd.DataFrame,
    features: list,
    filter_specs: list,
    target_col: str,
    output_target_col: str,
    spec: dict,
    dataset_label: str,
    row_filters: list,
    apply_outlier_filter: bool,
):
    df = drop_required_null_rows(raw_df, features, dataset_label)
    df = apply_feature_cleaning(df, features, dataset_label)
    df = apply_filter_spec_cleaning(df, filter_specs, dataset_label)
    df = apply_row_filters(df, row_filters, dataset_label)
    if target_col not in df.columns:
        _fail(f"{dataset_label} target column '{target_col}' is not present.")
    df = clean_target(
        df,
        target_col,
        spec,
        dataset_label=dataset_label,
        apply_outlier_filter=apply_outlier_filter,
    )
    if target_col != output_target_col:
        df = df.rename(columns={target_col: output_target_col})
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic data preparation pipeline")
    parser.add_argument(
        "--workspace",
        default=".",
        help=(
            "Workspace root containing local config/data directories "
            "(default: current directory)."
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()
    paths = WorkspacePaths.from_value(args.workspace)
    os.makedirs(paths.data_dir, exist_ok=True)

    spec = load_spec(paths)
    spec_label = paths.display_path(paths.feature_spec_path)
    features = spec["features"]
    filter_specs = spec.get("filter_specs", [])
    target = spec["target_column"]
    test_data_file = spec.get("test_data_file")
    test_target_column = spec.get("test_target_column", target)
    train_filters = spec.get("train_row_filters", [])
    test_filters = spec.get("test_row_filters", [])

    print(f"[prepare] Workspace     : {paths.display_workspace_root()}")
    print(f"[prepare] Spec version  : {spec.get('version', '?')}")
    print(f"[prepare] Train source  : {spec['data_file']}")
    print(f"[prepare] Target        : {target}")
    print(f"[prepare] Features      : {[f['column'] for f in features]}")
    if filter_specs:
        print(f"[prepare] Filter specs  : {[f['column'] for f in filter_specs]}")
    if train_filters:
        print(f"[prepare] Train filters : {train_filters}")
    if test_data_file:
        print(f"[prepare] Test source   : {test_data_file}")
    if test_filters:
        print(f"[prepare] Test filters  : {test_filters}")

    train_raw = load_table(spec["data_file"], spec_label=spec_label, paths=paths)
    train_prepared = prepare_dataset(
        train_raw,
        features,
        filter_specs,
        target_col=target,
        output_target_col=target,
        spec=spec,
        dataset_label="train source",
        row_filters=train_filters,
        apply_outlier_filter=True,
    )

    encoding_meta = fit_encoding_meta(train_prepared, features)
    train_encoded = apply_encodings(train_prepared, features, encoding_meta)
    train_final, model_cols, ref_cols = build_matrix(train_encoded, features, encoding_meta, target)

    if test_data_file:
        test_raw = load_table(test_data_file, spec_label=spec_label, paths=paths)
        test_prepared = prepare_dataset(
            test_raw,
            features,
            filter_specs,
            target_col=test_target_column,
            output_target_col=target,
            spec=spec,
            dataset_label="test source",
            row_filters=test_filters,
            apply_outlier_filter=False,
        )
        test_encoded = apply_encodings(test_prepared, features, encoding_meta)
        test_final, _unused_model_cols, _unused_ref_cols = build_matrix(
            test_encoded, features, encoding_meta, target
        )
        train_df = train_final.reset_index(drop=True)
        test_df = test_final.reset_index(drop=True)
        split_mode = "explicit_test_data_file"
    else:
        train_df, test_df = train_test_split(
            train_final,
            test_size=spec.get("test_size", 0.20),
            random_state=spec.get("random_state", 42),
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        split_mode = "split_from_train_source"

    print(f"\n[prepare] Final train shape: {train_df.shape[0]:,} rows")
    print(f"[prepare] Final test shape : {test_df.shape[0]:,} rows")
    print(f"[prepare] Model features   : {model_cols}")
    print(f"[prepare] Ref columns      : {ref_cols}")
    print(f"\n[prepare] Price/MSF stats (train prepared source):")
    print(train_final[target].describe().to_string())

    train_df.to_parquet(paths.train_data_path, index=False)
    test_df.to_parquet(paths.test_data_path, index=False)
    print(
        f"\n[prepare] Saved → {paths.display_path(paths.train_data_path)} "
        f"({len(train_df):,} rows)"
    )
    print(
        f"[prepare] Saved → {paths.display_path(paths.test_data_path)}   "
        f"({len(test_df):,} rows)"
    )

    meta = {
        "target": target,
        "model_features": model_cols,
        "ref_cols": ref_cols,
        "encoding": encoding_meta,
        "feature_config": features,
        "filter_spec_config": filter_specs,
        "domain_context": spec.get("domain_context", ""),
        "data_scope": {
            "train_source": spec["data_file"],
            "train_row_filters": train_filters,
            "test_source": test_data_file or spec["data_file"],
            "test_row_filters": test_filters,
            "split_mode": split_mode,
            "test_target_column": test_target_column,
        },
        "n_train": len(train_df),
        "n_test": len(test_df),
        "target_stats": {
            key: round(float(value), 4)
            for key, value in train_final[target].describe().items()
        },
    }
    meta["workspace_root"] = paths.display_workspace_root()
    with open(paths.meta_path, "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"[prepare] Saved → {paths.display_path(paths.meta_path)}")
    print(
        "\n[prepare] ✅ Done. Run: "
        f"python run_experiment.py init-train --workspace {paths.display_workspace_root()}"
    )


if __name__ == "__main__":
    main()
