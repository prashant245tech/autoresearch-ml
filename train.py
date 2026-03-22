"""
train.py - Editable experiment program for the current ML tuning task.

The external agent edits only this file while the deterministic runner in
`run_experiment.py` owns data splitting, validation scoring, and acceptance/export.

Required hooks:
1. `EXPERIMENT_DESCRIPTION`
2. `engineer_features(df, meta)`
3. `build_model(meta)`

Optional hooks:
4. `fit_model(model, X_train, y_train, X_val, y_val)`
5. `predict_model(model, X)`
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


EXPERIMENT_DESCRIPTION = (
    "move_intent=explore_new_branch | "
    "change_type=feature_probe | family=ExtraTrees | "
    "change=add catalog_x_size_bucket and catalog_x_ink_bucket on top of the current winner | "
    "hypothesis=separating catalog-style rows from live-order rows within size and print regimes may capture a distinct pricing pattern"
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

    # Base continuous features clipped properly
    sqft = None
    if "SQ. FT. PER PC" in df.columns:
        sqft = df["SQ. FT. PER PC"].clip(lower=0.01)
        fe["log_sqft"] = np.log1p(sqft)

    if "Quantity" in df.columns:
        qty = df["Quantity"].clip(lower=0)
        fe["qty_log"] = np.log1p(qty)

    # Encoded ordinals clipped non-negative (sentinel -1 already handled)
    size_bucket = None
    if "Size Bucket_encoded" in df.columns:
        size_bucket = df["Size Bucket_encoded"].clip(lower=0)

    ink_bucket = None
    if "Ink Coverage Bucket_encoded" in df.columns:
        ink_bucket = df["Ink Coverage Bucket_encoded"].clip(lower=0)

    # Add quantity tier categorical feature as ordinal bins of quantity
    if "Quantity" in df.columns:
        bins = [-1, 0, 500, 1000, 2500, 5000, 10000, np.inf]
        qty_tier = pd.cut(df["Quantity"], bins=bins, labels=False).fillna(-1).astype(int)
        fe["qty_tier"] = qty_tier.clip(lower=0)

    # Setup proxy for fixed cost amortization (inversely proportional to area * qty_log)
    if sqft is not None and "qty_log" in fe.columns:
        denom = sqft * fe["qty_log"] + 1.0
        fe["setup_proxy"] = 1.0 / denom
        fe["area_x_qty"] = sqft * fe["qty_log"]
        total_board = sqft * df["Quantity"].clip(lower=0)
        fe["log_total_board"] = np.log1p(total_board)

    # Cross features suggested by domain knowledge
    if sqft is not None and "Flute 1_encoded" in df.columns:
        flute = df["Flute 1_encoded"].clip(lower=0)
        fe["area_x_flute"] = sqft * flute
        if size_bucket is not None:
            fe["flute_x_size_bucket"] = flute * size_bucket
    if sqft is not None and ink_bucket is not None:
        fe["area_x_ink_bucket"] = sqft * ink_bucket
        if size_bucket is not None:
            fe["size_x_ink_bucket"] = size_bucket * ink_bucket
        if "Flute 1_encoded" in df.columns:
            flute = df["Flute 1_encoded"].clip(lower=0)
            fe["flute_x_ink_bucket"] = flute * ink_bucket
    if size_bucket is not None and "qty_log" in fe.columns and sqft is not None:
        fe["area_x_size_bucket"] = sqft * size_bucket
    if "qty_log" in fe.columns and ink_bucket is not None:
        fe["qty_x_ink_bucket"] = fe["qty_log"] * ink_bucket

    if "Tare Weight" in df.columns and sqft is not None:
        # Add log tare weight to capture diminishing returns of weight effect
        fe["log_tare_weight"] = np.log1p(df["Tare Weight"].clip(lower=0.001))

    # Add catalog price flag (has_quantity already added)
    if "Quantity" in df.columns:
        fe["is_catalog_price"] = (df["Quantity"] == 0).astype(int)
        if size_bucket is not None:
            fe["catalog_x_size_bucket"] = fe["is_catalog_price"] * size_bucket
        if ink_bucket is not None:
            fe["catalog_x_ink_bucket"] = fe["is_catalog_price"] * ink_bucket

    return fe


def build_model(meta: dict):
    """
    Return the actual estimator object for the immutable harness to fit.
    """
    return ExtraTreesRegressor(
        n_estimators=900,
        max_depth=20,
        min_samples_leaf=1,
        max_features=0.55,
        criterion="absolute_error",
        random_state=42,
        n_jobs=-1,
    )


def fit_model(model, X_train, y_train, X_val, y_val):
    """
    Optional hook for custom training behavior.

    Validation metrics are still computed in run_experiment.py.
    """
    model.fit(X_train, np.log1p(y_train))
    return model


def predict_model(model, X):
    """
    Optional hook for custom prediction behavior.
    """
    return np.expm1(model.predict(X))


if __name__ == "__main__":
    from run_experiment import run_single_from_train_entrypoint

    run_single_from_train_entrypoint()
