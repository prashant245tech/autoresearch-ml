"""
prepare.py — Generic data preparation pipeline.
Reads feature_spec.json produced by compile.py. No LLM calls here.
DO NOT MODIFY. The agent only modifies train.py.

Run order:
    python compile.py    ← translates program.md → feature_spec.json  (once)
    python prepare.py    ← executes spec → train/test parquet          (once per data change)
    python train.py      ← agent iterates on this
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

SPEC_PATH  = "feature_spec.json"
TRAIN_PATH = "data/train.parquet"
TEST_PATH  = "data/test.parquet"
META_PATH  = "data/columns.json"


def load_spec() -> dict:
    if not os.path.exists(SPEC_PATH):
        sys.exit(
            f"[prepare] ERROR: {SPEC_PATH} not found.\n"
            f"          Run: python compile.py"
        )
    with open(SPEC_PATH) as f:
        return json.load(f)


def load_table(file_path: str) -> pd.DataFrame:
    for p in [file_path,
              f"/mnt/user-data/uploads/{os.path.basename(file_path)}",
              f"data/{os.path.basename(file_path)}"]:
        if os.path.exists(p):
            ext = os.path.splitext(p)[1].lower()
            if ext in (".xlsx", ".xls"):
                df = pd.read_excel(p)
            else:
                df = pd.read_csv(p)
            print(f"[prepare] Loaded: {p}  →  {df.shape[0]:,} rows × {df.shape[1]} cols")
            return df
    sys.exit(
        f"[prepare] ERROR: File not found at '{file_path}'.\n"
        f"          Update data_file in program.md and re-run compile.py."
    )


def apply_feature_spec(df: pd.DataFrame, feat: dict) -> pd.DataFrame:
    col = feat["column"]
    if col not in df.columns:
        print(f"[prepare] WARNING: '{col}' not in input file — skipping")
        return df

    ftype         = feat.get("type", "numeric")
    normalise     = feat.get("normalise", "none")
    fill_cfg      = feat.get("fill_null", {})
    clip_cfg      = feat.get("clip", {})
    consolidation = feat.get("categorical_consolidation", {})

    # Normalise strings
    if normalise == "uppercase_strip":
        df[col] = (df[col].astype(str).str.strip().str.upper()
                          .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan}))

    # Consolidate categorical variants
    if consolidation:
        upper_map = {str(k).strip().upper(): str(v).strip().upper()
                     for k, v in consolidation.items()}
        df[col] = df[col].map(
            lambda x: upper_map.get(str(x).strip().upper(), x) if pd.notna(x) else x
        )

    # Fill nulls
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
        m = df[col].mode()
        if not m.empty:
            df[col] = df[col].fillna(m.iloc[0])

    # Type coercion
    if ftype in ("numeric", "boolean"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if ftype == "boolean":
            df[col] = df[col].fillna(0).clip(0, 1).astype(int)

    # Clip
    if ftype == "numeric":
        lo, hi = clip_cfg.get("min"), clip_cfg.get("max")
        if lo is not None:
            df[col] = df[col].clip(lower=lo)
        if hi is not None:
            df[col] = df[col].clip(upper=hi)

    return df


def encode_features(df: pd.DataFrame, features: list) -> tuple[pd.DataFrame, dict]:
    encoding_meta = {}

    for feat in features:
        col   = feat["column"]
        ftype = feat.get("type", "numeric")
        if col not in df.columns:
            continue

        if ftype == "ordinal":
            order    = [str(v).strip().upper() for v in feat.get("ordinal_order", [])]
            sentinel = feat.get("unknown_sentinel", -1)
            mapping  = {v: i+1 for i, v in enumerate(order)}
            enc_col  = f"{col}_encoded"
            raw_vals = (df[col].astype(str).str.strip().str.upper()
                               .replace({"NAN": np.nan, "NONE": np.nan}))
            df[f"{col}_raw"] = raw_vals.fillna("UNKNOWN")
            df[enc_col]      = raw_vals.map(mapping).fillna(sentinel).astype(int)
            encoding_meta[col] = {
                "type": "ordinal", "order": order,
                "encoded_col": enc_col, "sentinel": sentinel
            }

        elif ftype == "categorical":
            sentinel = str(feat.get("unknown_sentinel", "UNKNOWN"))
            enc_col  = f"{col}_enc"
            raw_vals = (df[col].astype(str).str.strip().str.upper()
                               .replace({"NAN": sentinel, "NONE": sentinel, "": sentinel}))
            df[f"{col}_raw"] = raw_vals
            le = LabelEncoder()
            df[enc_col] = le.fit_transform(raw_vals)
            encoding_meta[col] = {
                "type": "categorical", "encoded_col": enc_col,
                "classes": list(le.classes_)
            }

    return df, encoding_meta


def build_matrix(df, features, encoding_meta):
    model_cols, ref_cols = [], []
    for feat in features:
        col   = feat["column"]
        ftype = feat.get("type", "numeric")
        if col not in df.columns:
            continue
        if ftype in ("numeric", "boolean"):
            model_cols.append(col)
        elif ftype == "ordinal":
            enc = encoding_meta.get(col, {}).get("encoded_col", f"{col}_encoded")
            if enc in df.columns:
                model_cols.append(enc)
            raw = f"{col}_raw"
            if raw in df.columns:
                ref_cols.append(raw)
        elif ftype == "categorical":
            enc = encoding_meta.get(col, {}).get("encoded_col", f"{col}_enc")
            if enc in df.columns:
                model_cols.append(enc)
            raw = f"{col}_raw"
            if raw in df.columns:
                ref_cols.append(raw)

    all_cols = [c for c in model_cols + ref_cols + ["price_msf"] if c in df.columns]
    return df[all_cols].reset_index(drop=True), model_cols, ref_cols


def clean_target(df, target_col, spec):
    before = len(df)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].notna() & ~np.isinf(df[target_col]) & (df[target_col] > 0)]
    k = float(spec.get("outlier_iqr_k", 4.0))
    if k:
        q1, q3 = df[target_col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[(df[target_col] >= q1 - k * iqr) & (df[target_col] <= q3 + k * iqr)]
    print(f"[prepare] Target cleaned : {before:,} → {len(df):,} rows "
          f"({before - len(df):,} removed)")
    return df


def main():
    os.makedirs("data", exist_ok=True)

    spec     = load_spec()
    features = spec["features"]
    target   = spec["target_column"]

    print(f"[prepare] Spec version  : {spec.get('version', '?')}")
    print(f"[prepare] Target        : {target}")
    print(f"[prepare] Features      : {[f['column'] for f in features]}")

    df = load_table(spec["data_file"])

    # Drop required-but-null rows
    drop_cols = [f["column"] for f in features
                 if f.get("drop_if_null") and f["column"] in df.columns]
    if drop_cols:
        before = len(df)
        df = df.dropna(subset=drop_cols)
        print(f"[prepare] Dropped {before - len(df):,} rows with nulls in: {drop_cols}")

    # Apply per-feature cleaning
    for feat in features:
        df = apply_feature_spec(df, feat)

    # Clean target
    df = clean_target(df, target, spec)
    df = df.rename(columns={target: "price_msf"})

    # Encode
    df, encoding_meta = encode_features(df, features)

    # Build matrix
    df_final, model_cols, ref_cols = build_matrix(df, features, encoding_meta)

    print(f"\n[prepare] Final shape   : {df_final.shape[0]:,} rows")
    print(f"[prepare] Model features: {model_cols}")
    print(f"[prepare] Ref columns   : {ref_cols}")
    print(f"\n[prepare] Price/MSF stats:")
    print(df_final["price_msf"].describe().to_string())

    # Split
    train_df, test_df = train_test_split(
        df_final,
        test_size=spec.get("test_size", 0.20),
        random_state=spec.get("random_state", 42)
    )
    train_df.to_parquet(TRAIN_PATH, index=False)
    test_df.to_parquet(TEST_PATH,   index=False)
    print(f"\n[prepare] Saved → {TRAIN_PATH} ({len(train_df):,} rows)")
    print(f"[prepare] Saved → {TEST_PATH}   ({len(test_df):,} rows)")

    meta = {
        "target":          "price_msf",
        "model_features":  model_cols,
        "ref_cols":        ref_cols,
        "encoding":        encoding_meta,
        "feature_config":  features,
        "domain_context":  spec.get("domain_context", ""),
        "n_train":         len(train_df),
        "n_test":          len(test_df),
        "price_msf_stats": {
            k: round(float(v), 4)
            for k, v in df_final["price_msf"].describe().items()
        },
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[prepare] Saved → {META_PATH}")
    print(f"\n[prepare] ✅ Done. Run: python train.py")


if __name__ == "__main__":
    main()
