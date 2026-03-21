# program.md - External agent workflow and generic ML tuning policy

This file guides the external agent that edits `train.py`.
It is not consumed by `run_experiment.py`.

## External Agent Loop

### Setup phase

If the prepared parquet files do not exist yet, or if the dataset definition has changed:

1. Update the task-specific `config` and `features` sections in this file.
2. Create or update `feature_spec.json` directly from those sections.
   - Keep `feature_spec.json` explicit and machine-readable because `prepare.py`
     executes it deterministically.
   - Follow `feature_spec.schema.json` when writing the spec.
   - If the user wants to train on a subset of rows, encode that as
     `train_row_filters` in `feature_spec.json`.
   - If a filtered column should be cleaned or normalized but should not become
     a model feature, declare it in `filter_specs`.
   - If the user wants a separate evaluation dataset, set `test_data_file`
     and optional `test_target_column` in `feature_spec.json`.
   - If the user wants to scope the explicit test dataset too, encode that as
     `test_row_filters` in `feature_spec.json`.
3. Run `python prepare.py`.
4. Delete `experiments/session_baseline.json` if it exists from a prior session.
5. Freeze the prepared data for the tuning session. The first
   `python run_experiment.py run` or `accept` call will create a new
   session-baseline file that pins the prepared-data hashes.

### Tuning phase

1. Read this file before proposing a change.
2. Edit `train.py` only.
3. Run `python run_experiment.py run`.
4. Compare the returned validation metrics against your own accepted-run history.
5. Keep or discard the `train.py` change externally.
6. If you reran `prepare.py` since the last accepted run, delete
   `experiments/session_baseline.json` before continuing so the new tuning
   session gets a clean prepared-data baseline.
7. If you keep it, run:

```bash
python run_experiment.py accept --expected-train-sha <train_sha>
```

8. Commit or discard externally. The repo itself does not manage git, experiment
   history, or best-model state.

## Fixed Protocol

- `prepare.py` defines the frozen `data/train.parquet` and `data/test.parquet` boundary.
- `experiments/session_baseline.json` enforces that the prepared-data hashes stay fixed
  for the tuning session.
- `run_experiment.py run` creates the deterministic internal validation split.
- Validation metrics are the search signal.
- Test metrics are audit-only and are produced during `accept`.
- The fixed experiment budget for `run` is 300 seconds.
- The saved accepted artifact is retrained on all of `data/train.parquet`.

## Editable Surface

Edit only `train.py`, specifically:
- `EXPERIMENT_DESCRIPTION`
- `engineer_features(df, meta)`
- `build_model(meta)`
- Optional: `fit_model(model, X_train, y_train, X_val, y_val)`
- Optional: `predict_model(model, X)`

The runner owns splitting, metric computation, and artifact export.

## Run Telemetry

`python run_experiment.py run` now returns enough signal for generic diagnosis:
- `train_mape`, `train_rmse`, `train_r2`
- `val_mape`, `val_rmse`, `val_r2`
- `train_val_mape_gap`, `train_val_mape_ratio`, `train_val_rmse_gap`
- `fit_elapsed_s`, `predict_train_elapsed_s`, `predict_val_elapsed_s`
- `elapsed_budget_fraction`
- `model_class`, `model_params`
- `n_train_rows`, `n_val_rows`
- `n_features`, `n_base_features`, `n_derived_features`
- `base_feature_names`, `derived_feature_names`, `omitted_base_feature_names`

## Generic Edit Policy

### Core rules

- Make one coherent change per run.
- Prefer reversible, readable edits over large rewrites.
- Search on validation metrics only. Never choose experiments based on test results.
- Use `EXPERIMENT_DESCRIPTION` to state the change type, the concrete edit, and the hypothesis.

Recommended format:

```python
EXPERIMENT_DESCRIPTION = (
    "change_type=param_refine | family=HistGradientBoosting | "
    "change=max_depth 8->12 | "
    "hypothesis=reduce underfitting on nonlinear interactions"
)
```

### Default search order

1. Establish a working baseline if none exists.
2. Probe several model families with mostly stable features.
3. Once one family clearly leads, spend a few runs on local hyperparameter refinement.
4. When model tuning plateaus, try one feature block change.
5. After an accepted feature improvement, retune the current best model family once.

### Suggested change types

- `family_probe`
  - Change model family only.
  - Keep the current feature set stable.
- `param_refine`
  - Keep the model family fixed.
  - Adjust a small number of hyperparameters around the current best setting.
- `feature_probe`
  - Add, remove, or revise one coherent feature block.
  - Keep the model family and most hyperparameters fixed.
- `feature_cleanup`
  - Remove noisy or redundant features after several weak feature additions.
  - Keep the rest of the experiment stable.

### Switching criteria

- If multiple family probes have not been tried yet, prioritize family exploration before deep tuning.
- If one family is clearly ahead, stay within that family for a short local sweep before switching again.
- If 3 consecutive parameter tweaks fail to improve validation metrics, switch to a different family or a feature change.
- If several families are very close, prefer the simpler or faster one.
- If runtime approaches the budget, favor cheaper models or smaller configurations.
- If `train_val_mape_gap` is large or `train_val_mape_ratio` is well above 1.0, simplify or regularize.
- If both train and validation metrics are poor, increase model capacity or improve the feature set.
- If train metrics are strong but validation is weak, prefer regularization, shallower trees, larger leaves, or fewer derived interactions.
- If runtime rises materially without validation gain, prefer the cheaper model or smaller configuration.
- If the derived-feature count grows while validation stalls, try feature cleanup before more model tuning.

### Practical hyperparameter guidance

- For bagged tree models, tune capacity and leaf-size controls before increasing estimator count.
- For boosting models, tune learning-rate and tree-size controls before increasing rounds.
- For linear models, use them mainly as baselines or when they remain competitive.
- Avoid changing both model family and feature engineering in the same run.

### Fast diagnosis playbook

- `low train error + much worse validation error`
  - overfitting signal
  - simplify the model, raise regularization, or remove noisy derived features
- `high train error + high validation error`
  - underfitting signal
  - try a stronger family, deeper trees, smaller leaves, or richer features
- `small accuracy gap between two runs but large runtime gap`
  - prefer the cheaper run
- `many omitted base features`
  - inspect whether `engineer_features()` is unintentionally dropping useful prepared inputs
- `derived features dominate with no gain`
  - run a cleanup or family probe instead of stacking more interactions

## Task-Specific Context

This section is specific to the current corrugated pricing dataset. It is useful
for the external agent when writing `feature_spec.json` during setup.

## Dataset Config

```config
data_file: /Users/prashant/Downloads/Tier_TreeHouse_Winland 1 - Copy.xlsx
target_column: Tier 1 Price/MSF
test_size: 0.20
random_state: 42
outlier_iqr_k: 4.0
```

If you want a separate holdout dataset, add `test_data_file` and optional
`test_target_column` to `feature_spec.json` directly.

## Active Cohort Setup

- Current requested cohort: `Client = TreeHouse`
- Current requested price scope: `Price Category = MODEL`
- Train/test behavior: split train and test from this same filtered cohort
- Exclude: `LEGACY`, `OFF MODEL`, and `NEEDS REVIEW`

## Optional Prep Controls

Use these when the task requires cohort-specific training or an explicit test set.
The external agent should translate them into `feature_spec.json`.

Example:

```json
{
  "filter_specs": [
    {
      "column": "Price Category",
      "type": "categorical",
      "normalise": "uppercase_strip",
      "fill_null": {"strategy": "value", "value": "UNKNOWN"},
      "categorical_consolidation": {
        "LEGACY PRICE": "LEGACY",
        "LEGACY": "LEGACY"
      },
      "unknown_sentinel": "UNKNOWN"
    },
    {
      "column": "Account Type",
      "type": "categorical",
      "normalise": "uppercase_strip",
      "fill_null": {"strategy": "value", "value": "UNKNOWN"},
      "categorical_consolidation": {
        "FREE HOUSE ": "FREE HOUSE",
        "FREEHOUSE": "FREE HOUSE"
      },
      "unknown_sentinel": "UNKNOWN"
    }
  ],
  "train_row_filters": [
    {"column": "Price Category", "op": "in", "value": ["LEGACY"]},
    {"column": "Account Type", "op": "eq", "value": "FREE HOUSE"}
  ],
  "test_data_file": "/absolute/path/to/test_dataset.xlsx",
  "test_target_column": "Tier 1 Price/MSF",
  "test_row_filters": [
    {"column": "Account Type", "op": "eq", "value": "FREE HOUSE"}
  ]
}
```

Supported filter ops:
- `eq`, `ne`
- `in`, `not_in`
- `gt`, `gte`, `lt`, `lte`
- `contains`
- `is_null`, `not_null`

## Feature Spec

```features

Size Bucket:
  Pre-bucketed size band for the box based on dimensional scale.
  Ordered from smallest to largest: 0-3.0, 3.1-5.5, 5.6-8.5, 8.6-15.0, 15.1-30.0, 30.1-90.0.
  Treat unexpected or null values as unknown sentinel -1.

SQ. FT. PER PC:
  Surface area of the corrugated blank in square feet per piece.
  The single most important pricing driver - more area means more board consumed.
  Fill missing with column median. Clip to minimum 0.01.

Quantity:
  Order quantity in pieces. Many rows have zero - these are catalog or model prices,
  not actual orders. Keep zero rows but the distinction matters for the model.
  Fill missing with 0. Clip negative values to 0.

Flute 1:
  Flute type - an ordered ranking by board thickness and cost.
  Order from thinnest/cheapest to thickest/most expensive: F, E, N, B, C, BC.
  BC is double-wall and costs roughly 40-60% more per MSF than B-flute.
  Normalise inconsistent casing and spaces. Unknown values get a sentinel of -1.

Stock:
  Liner type. Either KRAFT (natural brown, cheaper) or WHITE (bleached, more expensive).
  Normalise messy casing variants (wHITE, white -> WHITE). Anything not clearly
  KRAFT or WHITE should be treated as OTHER. Fill missing with UNKNOWN.

Box Style:
  Structural style of the box. Main families: RSC, DCT, DCJ, TRAY, HSC, SHT, PAD.
  Consolidate messy variants: D/C JOINED -> DCJ, D/C NON JOINED -> DCJ,
  Die Cut -> DCT, D/C RSC -> RSC, D/C HSC -> HSC. Unknown -> OTHER.
  Fill missing with UNKNOWN.

Region:
  Geographic sales region - Midwest, Northeast, Southeast, Southwest, West, Canada.
  Affects price due to freight costs and regional supplier pricing differences.
  Normalise case. Fill missing with UNKNOWN.

Ink Coverage Bucket:
  Pre-bucketed print coverage level ordered from least to most ink:
  0-10%, 10-20%, 20-30%, 30-40%, 40-50%, >50%.
  Treat Null or unexpected values as unknown sentinel -1.

Tare Weight:
  Weight of the empty box in pounds. Correlates with board weight and caliper.
  Fill missing with column median. Clip to minimum 0.001.

Adhesive:
  Adhesive type used in box manufacture. MRA is standard, REGULAR is basic,
  WPA is a specialty type. Normalise casing. Fill missing with UNKNOWN.

Litho Box:
  Binary flag - 1 if this is a lithographic (premium printed) box, 0 otherwise.
  Litho boxes command a significant price premium. Fill missing with 0.

```

## Optional Task Notes

Use this section for domain-specific guidance the external agent should consider.

### What we are predicting
- **Tier 1 Price/MSF** is price per thousand square feet of corrugated board
  for the primary tiered quote level.

### Dataset summary
- Source: TreeHouse Foods / Winland supplier pricing data
- About 2,950 usable rows after cleaning
- Mix of RSC, DCT, DCJ, TRAY, HSC box styles
- Predominantly B-flute and C-flute with a small BC double-wall segment

### Likely drivers
1. Board area (`SQ. FT. PER PC`)
2. Quantity and price-break behavior
3. Size bucket
4. Flute type
5. Printing (`Ink Coverage Bucket`, `Litho Box`)
6. Stock
7. Box style
8. Region

### Constraints
- Do not use the target column as a feature.
- Zero-quantity rows are valid but should be treated differently from real orders.
- Keep experiments within the fixed 300-second `run` budget.
- Favor feature engineering that reflects pricing logic such as area, quantity breaks,
  setup amortization, and print complexity.

### Derived features worth trying

```python
log_sqft = np.log1p(df["SQ. FT. PER PC"])
qty_log = np.log1p(df["Quantity"])
has_quantity = (df["Quantity"] > 0).astype(int)
area_x_qty = df["SQ. FT. PER PC"] * qty_log
area_x_flute = df["SQ. FT. PER PC"] * df["Flute 1_encoded"]
area_x_size = df["SQ. FT. PER PC"] * df["Size Bucket_encoded"]
area_x_ink = df["SQ. FT. PER PC"] * df["Ink Coverage Bucket_encoded"]
weight_per_sqft = df["Tare Weight"] / (df["SQ. FT. PER PC"] + 1e-6)
setup_proxy = 1.0 / (df["SQ. FT. PER PC"] * qty_log + 1.0)
qty_tier = pd.cut(
    df["Quantity"],
    bins=[-1, 0, 500, 1000, 2500, 5000, 10000, np.inf],
    labels=False,
)
```
