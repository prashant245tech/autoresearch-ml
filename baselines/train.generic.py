"""
Neutral baseline train spec for local bootstrap.

Copy this file to `train.py` with:

    python run_experiment.py init-train

Then edit the generated local `train.py` for the active task.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


EXPERIMENT_DESCRIPTION = (
    "move_intent=explore_new_branch | "
    "change_type=family_probe | family=RandomForest | "
    "change=neutral pass-through baseline | "
    "hypothesis=a simple generic tree baseline provides a stable starting point"
)


def engineer_features(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Keep the prepared numeric model features as the starting hypothesis.
    """
    features = pd.DataFrame(index=df.index)
    for col in meta.get("model_features", []):
        if col in df.columns:
            features[col] = df[col]
    return features


def build_model(meta: dict):
    """
    A neutral, deterministic tabular baseline for local experimentation.
    """
    return RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


if __name__ == "__main__":
    from run_experiment import run_single_from_train_entrypoint

    run_single_from_train_entrypoint()
