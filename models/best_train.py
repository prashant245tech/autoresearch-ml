"""
train.py — Research spec for corrugated box price prediction.

Edit only the mutable research surface below:
1. `EXPERIMENT_DESCRIPTION`
2. `engineer_features(df, meta)`
3. `get_model_config()`

The immutable execution harness, evaluation rules, logging, and git workflow
live in `run_experiment.py`.
"""

import numpy as np
import pandas as pd


EXPERIMENT_DESCRIPTION = (
    "Model experiment: baseline bucket features with Extra Trees ensemble"
)


def engineer_features(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Build a numeric feature matrix from the prepared parquet inputs.

    Rules:
    - Return numeric columns only
    - Do not use `price_msf` as a feature
    - Check that columns exist before using them
    - `meta["model_features"]` contains the prepared base features
    """
    fe = pd.DataFrame(index=df.index)

    for col in meta.get("model_features", []):
        if col in df.columns:
            fe[col] = df[col]

    if "SQ. FT. PER PC" in df.columns:
        sqft = df["SQ. FT. PER PC"].clip(lower=0.01)
        fe["sqft"] = sqft
        fe["log_sqft"] = np.log1p(sqft)

    if "Quantity" in df.columns:
        qty = df["Quantity"].clip(lower=0)
        fe["qty_log"] = np.log1p(qty)
        fe["has_quantity"] = (qty > 0).astype(int)

    if "Size Bucket_encoded" in df.columns:
        size_bucket = df["Size Bucket_encoded"].clip(lower=0)
        fe["size_bucket"] = size_bucket

    if "Ink Coverage Bucket_encoded" in df.columns:
        ink_bucket = df["Ink Coverage Bucket_encoded"].clip(lower=0)
        fe["ink_bucket"] = ink_bucket

    if "sqft" in fe.columns and "qty_log" in fe.columns:
        fe["area_x_qty"] = fe["sqft"] * fe["qty_log"]

    if "sqft" in fe.columns and "Flute 1_encoded" in df.columns:
        fe["area_x_flute"] = fe["sqft"] * df["Flute 1_encoded"].clip(lower=0)

    if "sqft" in fe.columns and "ink_bucket" in fe.columns:
        fe["area_x_ink_bucket"] = fe["sqft"] * fe["ink_bucket"]

    if "sqft" in fe.columns and "size_bucket" in fe.columns:
        fe["area_x_size_bucket"] = fe["sqft"] * fe["size_bucket"]

    if "qty_log" in fe.columns and "size_bucket" in fe.columns:
        fe["qty_x_size_bucket"] = fe["qty_log"] * fe["size_bucket"]

    if "qty_log" in fe.columns and "ink_bucket" in fe.columns:
        fe["qty_x_ink_bucket"] = fe["qty_log"] * fe["ink_bucket"]

    if "Tare Weight" in df.columns and "sqft" in fe.columns:
        fe["weight_per_sqft"] = df["Tare Weight"] / (fe["sqft"] + 1e-6)

    return fe


def get_model_config() -> dict:
    """
    Return a declarative model config for the immutable harness.

    Supported families:
    - xgboost
    - lightgbm
    - ridge
    - lasso
    - elasticnet
    - random_forest
    - extra_trees
    - gradient_boosting
    """
    return {
        "family": "extra_trees",
        "params": {
            "n_estimators": 500,
            "max_depth": 16,
            "min_samples_leaf": 2,
            "max_features": 0.8,
        },
    }


if __name__ == "__main__":
    from run_experiment import run_single_from_train_entrypoint

    run_single_from_train_entrypoint()
