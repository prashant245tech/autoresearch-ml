# AutoResearch-Style ML Tuning Workflow

This repo uses a thin deterministic harness around an editable workspace-local
`train.py`. The external agent owns the research loop. The repo does not own
autonomous LLM control, best-model state, or git keep/discard behavior.

## Core Files

```text
autoresearch-ml/
|- program.md        external-agent tuning policy
|- workspace_paths.py shared workspace path resolution
|- baselines/
|  `- train.generic.py tracked neutral baseline for bootstrapping workspace train.py
|- run_experiment.py deterministic runner with workspace-aware `run`, `accept`, and `memory-summary`
|- search_memory.py  local JSONL search-memory storage and summary logic
|- explain.py        optional SHAP post-hoc analysis for accepted artifacts
|- prepare.py        fixed data preparation -> <workspace>/data/train.parquet + test.parquet
|- config/
|  `- feature_spec.schema.json JSON schema for the prep spec
`- workspaces/
   `- <workspace>/
      |- config/
      |  |- feature_spec.json agent-written prep spec consumed by prepare.py
      |  `- task_context.md local task-specific notes (optional, untracked)
      |- data/
      |  |- train.parquet
      |  |- test.parquet
      |  `- columns.json
      |- experiments/
      |  |- session_baseline.json
      |  |- search_memory.jsonl
      |  `- search_summary.json
      |- models/
      |  `- accepted/<train_sha[:12]>/
      `- train.py local generated working copy (untracked)
```

## Design

- workspace-local `train.py` owns the hypothesis:
  - `EXPERIMENT_DESCRIPTION`
  - `engineer_features(df, meta)`
  - `build_model(meta)`
  - optional `fit_model(...)`
  - optional `predict_model(...)`
- `run_experiment.py` owns the protocol:
  - fixed train/validation split from `<workspace>/data/train.parquet`
  - validation scoring
  - fixed 300-second budget for `run`
  - accepted-model export
  - local search-memory recording and summary retrieval
- `explain.py` owns optional post-hoc interpretation for accepted artifacts.
- `program.md` tells the external agent how to iterate.
- `program.md` also contains the default machine-readable policy thresholds for search behavior.

## Workflow

### 1. Choose a workspace

Pick or create a workspace directory, for example:

```bash
WORKSPACE=workspaces/my_dataset
```

### 2. Prepare data when the dataset changes

If you already have `$WORKSPACE/data/train.parquet`, `$WORKSPACE/data/test.parquet`,
and `$WORKSPACE/data/columns.json`,
you can skip this section.

1. Update `$WORKSPACE/config/task_context.md` if the current task needs local dataset-specific notes.
2. Create or update `$WORKSPACE/config/feature_spec.json` from the local task context and user instructions.
   - Use `train_row_filters` to restrict the training cohort.
   - Use `filter_specs` when filters depend on cleanup rules for columns that are not model features.
   - Use `test_data_file` and optional `test_target_column` for an explicit holdout dataset.
   - Use `test_row_filters` to scope the explicit holdout dataset when needed.
   - Follow [feature_spec.schema.json](/Users/prashant/Downloads/corrugated_automl 2/config/feature_spec.schema.json).
3. Run:

```bash
python prepare.py --workspace "$WORKSPACE"
```

The setup contract is:
- `program.md` defines the generic agent workflow
- `$WORKSPACE/config/task_context.md` holds local task-specific notes when needed
- `$WORKSPACE/config/feature_spec.json` is the executable prep spec
- `prepare.py` consumes that explicit spec deterministically

The first `run` or `accept` in a tuning session creates
`$WORKSPACE/experiments/session_baseline.json`, which records the hashes of the prepared
data files. Future runs must match that baseline exactly. If you rerun
`prepare.py` or otherwise change the prepared data, delete
`$WORKSPACE/experiments/session_baseline.json` before starting the next tuning session.

### 3. Edit only `train.py`

Bootstrap a local editable `train.py` first if it does not exist yet:

```bash
python run_experiment.py init-train --workspace "$WORKSPACE"
```

This copies the tracked neutral baseline from
[baselines/train.generic.py](/Users/prashant/Downloads/corrugated_automl%202/baselines/train.generic.py)
into a local untracked `$WORKSPACE/train.py`.

The external agent then reads `program.md`, checks local search memory, and modifies `train.py`.

```bash
python run_experiment.py memory-summary --workspace "$WORKSPACE"
```

### 4. Run one experiment

```bash
python run_experiment.py run --workspace "$WORKSPACE"
```

This prints a machine-readable JSON summary with:
- `train_sha`
- `experiment_description`
- `train_mape`, `train_rmse`, `train_r2`
- `val_mape`, `val_rmse`, `val_r2`
- `train_val_mape_gap`, `train_val_mape_ratio`
- `fit_elapsed_s`, `predict_train_elapsed_s`, `predict_val_elapsed_s`
- `model_class`, `model_params`
- `feature_importance_source`, `top_feature_importances` when the fitted model exposes native importances
- `n_features`, `n_base_features`, `n_derived_features`
- data fingerprints
- `session_baseline_path`
- runtime metadata

Each `run` also appends a local search-memory event to:
- `$WORKSPACE/experiments/search_memory.jsonl`
- `$WORKSPACE/experiments/search_summary.json`

Use `memory-summary` to avoid repeating the same `candidate_signature` or `train_sha`
unless you intentionally want a replicate run.

The search-memory summary also surfaces controller-friendly signals such as:
- `consecutive_exploit_count`
- `family_loss_streaks`
- `last_improvement_delta`
- `plateau_signal`
- `overfit_signal`

### 5. Accept and save a model

If the external agent decides to keep the current `train.py`, run:

```bash
python run_experiment.py accept --workspace "$WORKSPACE" --expected-train-sha <train_sha>
```

This:
- verifies the saved artifact matches the evaluated `train.py`
- verifies the prepared data still matches the session baseline
- retrains on all of `$WORKSPACE/data/train.parquet`
- evaluates audit metrics on `$WORKSPACE/data/test.parquet`
- saves an artifact bundle under `$WORKSPACE/models/accepted/<train_sha[:12]>`
- appends a successful `accept` event to local search memory

Before `accept`, apply the default confirmation rule:
- if the new candidate improves the current best `val_mape` by less than `0.10`
  absolute MAPE points, run one confirmation step first instead of accepting immediately

If the default artifact directory already exists, rerun `accept` with a unique
`--output-dir`. Treat that as workflow noise, not as model evidence.

Bundle contents:
- `model.pkl`
- `feature_columns.json`
- `train.py`
- `manifest.json`

### 6. Optional SHAP analysis on an accepted artifact

For a human-facing post-hoc explanation pass on an accepted artifact:

```bash
python explain.py --workspace "$WORKSPACE" --artifact-dir models/accepted/<artifact-id> --dataset test
```

This:
- loads the accepted `model.pkl`, `feature_columns.json`, and saved `train.py`
- rebuilds the feature matrix on the requested prepared-data split
- runs SHAP `TreeExplainer` if `shap` is installed
- writes a compact JSON summary under `$WORKSPACE/models/accepted/<artifact-id>/explanations/shap_summary.json`

Notes:
- `explain.py` is opt-in and is not part of the critical search path.
- It is primarily for human review, not for every-run agent control.
- If `shap` is not installed, `explain.py` exits cleanly with instructions to install it:

```bash
python3 -m pip install shap
```

## What The Repo Does Not Do

- no internal LLM loop
- no git commit or reset logic
- no automatic next-step search controller
- no repo-owned "best model" state

Those responsibilities belong to the external agent or your surrounding workflow.

## Notes

- Validation metrics are the search signal.
- Test metrics are audit-only and appear during `accept`.
- `prepare.py` remains the frozen train/test boundary inside each workspace.
- `prepare.py` can either split one source dataset or consume an explicit test dataset.
- The accepted artifact is retrained on all non-test training data.
- `<workspace>/experiments/session_baseline.json` is the enforced freeze-point for the prepared data.
- `<workspace>/experiments/search_memory.jsonl` is the local append-only search log.
- `<workspace>/experiments/search_summary.json` is the current prepared-data-scoped summary derived from that log.

## Tests

Run the framework tests with:

```bash
python3 -m unittest discover -s tests -v
```

The suite uses temporary workspaces, so it does not touch your local
workspace config, prepared parquet files, or accepted-model artifacts.
