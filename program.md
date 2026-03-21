# program.md - External agent workflow and generic ML tuning policy

This file guides the external agent that edits `train.py`.
It is not consumed by `run_experiment.py`.

## External Agent Loop

### Setup phase

If the prepared parquet files do not exist yet, or if the dataset definition has changed:

1. Update `config/task_context.md` if you need local dataset-specific notes for the current task.
2. Create or update `config/feature_spec.json` directly from the local task context and user instructions.
   - Keep `config/feature_spec.json` explicit and machine-readable because `prepare.py`
     executes it deterministically.
   - Follow `config/feature_spec.schema.json` when writing the spec.
   - If the user wants to train on a subset of rows, encode that as
     `train_row_filters` in `config/feature_spec.json`.
   - If a filtered column should be cleaned or normalized but should not become
     a model feature, declare it in `filter_specs`.
   - If the user wants a separate evaluation dataset, set `test_data_file`
     and optional `test_target_column` in `config/feature_spec.json`.
   - If the user wants to scope the explicit test dataset too, encode that as
     `test_row_filters` in `config/feature_spec.json`.
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

Keep this file generic. Store real dataset paths, cohort notes, domain hints,
and task-specific feature descriptions in the local untracked file
`config/task_context.md`.

Use `config/task_context.md` for:
- dataset paths and target column names
- active cohort filters or business-scope constraints
- task-specific feature descriptions
- domain notes the agent should consider during setup

If you want a separate holdout dataset, add `test_data_file` and optional
`test_target_column` to `config/feature_spec.json` directly.

Supported filter ops:
- `eq`, `ne`
- `in`, `not_in`
- `gt`, `gte`, `lt`, `lte`
- `contains`
- `is_null`, `not_null`
