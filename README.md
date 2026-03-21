# AutoResearch-Style ML Tuning Workflow

This repo uses a thin deterministic harness around an editable `train.py`.
The external agent owns the research loop. The repo does not own autonomous
LLM control, best-model state, or git keep/discard behavior.

## Core Files

```text
corrugated_automl/
|- program.md        external-agent tuning policy plus task-specific context
|- train.py          the only experiment file the agent edits
|- run_experiment.py deterministic runner with `run` and `accept`
|- prepare.py        fixed data preparation -> data/train.parquet + data/test.parquet
|- feature_spec.json agent-written prep spec consumed by prepare.py
|- feature_spec.schema.json JSON schema for the prep spec
|- data/
|  |- train.parquet
|  |- test.parquet
|  `- columns.json
`- models/
   `- accepted/<train_sha[:12]>/
```

## Design

- `train.py` owns the hypothesis:
  - `EXPERIMENT_DESCRIPTION`
  - `engineer_features(df, meta)`
  - `build_model(meta)`
  - optional `fit_model(...)`
  - optional `predict_model(...)`
- `run_experiment.py` owns the protocol:
  - fixed train/validation split from `data/train.parquet`
  - validation scoring
  - fixed 300-second budget for `run`
  - accepted-model export
- `program.md` tells the external agent how to iterate.

## Workflow

### 1. Prepare data when the dataset changes

If you already have `data/train.parquet`, `data/test.parquet`, and `data/columns.json`,
you can skip this section.

1. Customize the task-specific template in `program.md`.
2. Create or update `feature_spec.json` from those `config` and `features` sections.
   - Use `train_row_filters` to restrict the training cohort.
   - Use `filter_specs` when filters depend on cleanup rules for columns that are not model features.
   - Use `test_data_file` and optional `test_target_column` for an explicit holdout dataset.
   - Use `test_row_filters` to scope the explicit holdout dataset when needed.
   - Follow [feature_spec.schema.json](/Users/prashant/Downloads/corrugated_automl%202/feature_spec.schema.json).
3. Run:

```bash
python prepare.py
```

The setup contract is: `program.md` expresses intent, `feature_spec.json` makes
that intent explicit, and `prepare.py` consumes the explicit spec deterministically.

The first `run` or `accept` in a tuning session creates
`experiments/session_baseline.json`, which records the hashes of the prepared
data files. Future runs must match that baseline exactly. If you rerun
`prepare.py` or otherwise change the prepared data, delete
`experiments/session_baseline.json` before starting the next tuning session.

### 2. Edit only `train.py`

The external agent reads `program.md` and modifies `train.py`.

### 3. Run one experiment

```bash
python run_experiment.py run
```

This prints a machine-readable JSON summary with:
- `train_sha`
- `experiment_description`
- `train_mape`, `train_rmse`, `train_r2`
- `val_mape`, `val_rmse`, `val_r2`
- `train_val_mape_gap`, `train_val_mape_ratio`
- `fit_elapsed_s`, `predict_train_elapsed_s`, `predict_val_elapsed_s`
- `model_class`, `model_params`
- `n_features`, `n_base_features`, `n_derived_features`
- data fingerprints
- `session_baseline_path`
- runtime metadata

`python train.py` is a convenience alias for the same validation-only run.

### 4. Accept and save a model

If the external agent decides to keep the current `train.py`, run:

```bash
python run_experiment.py accept --expected-train-sha <train_sha>
```

This:
- verifies the saved artifact matches the evaluated `train.py`
- verifies the prepared data still matches the session baseline
- retrains on all of `data/train.parquet`
- evaluates audit metrics on `data/test.parquet`
- saves an artifact bundle under `models/accepted/<train_sha[:12]>`

Bundle contents:
- `model.pkl`
- `feature_columns.json`
- `train.py`
- `manifest.json`

## What The Repo Does Not Do

- no internal LLM loop
- no git commit or reset logic
- no `results.json` / `results.csv` leaderboard management
- no repo-owned "best model" state

Those responsibilities belong to the external agent or your surrounding workflow.

## Notes

- Validation metrics are the search signal.
- Test metrics are audit-only and appear during `accept`.
- `prepare.py` remains the frozen train/test boundary.
- `prepare.py` can either split one source dataset or consume an explicit test dataset.
- The accepted artifact is retrained on all non-test training data.
- `experiments/session_baseline.json` is the enforced freeze-point for the prepared data.
