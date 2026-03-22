# program.md - External agent workflow and generic ML tuning policy

This file guides the external agent that edits the local generated `train.py`.
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
2. If `train.py` does not exist yet, run `python run_experiment.py init-train`.
3. Read `python run_experiment.py memory-summary` to understand prior runs for the current prepared data.
4. Edit `train.py` only.
5. Run `python run_experiment.py run`.
6. Compare the returned validation metrics against search memory for the current prepared data.
7. Keep or discard the `train.py` change externally.
8. If you reran `prepare.py` since the last accepted run, delete
   `experiments/session_baseline.json` before continuing so the new tuning
   session gets a clean prepared-data baseline.
9. If you keep it, run:

```bash
python run_experiment.py accept --expected-train-sha <train_sha>
```

10. Commit or discard externally. The repo itself does not manage git, autonomous
   search control, or best-model state.

## Fixed Protocol

- `prepare.py` defines the frozen `data/train.parquet` and `data/test.parquet` boundary.
- `experiments/session_baseline.json` enforces that the prepared-data hashes stay fixed
  for the tuning session.
- `experiments/search_memory.jsonl` records run/accept events automatically.
- `experiments/search_summary.json` is the current prepared-data-scoped summary.
- `run_experiment.py run` creates the deterministic internal validation split.
- `run_experiment.py memory-summary` prints the current prepared-data-scoped search summary.
- `explain.py` is an optional post-hoc interpretation tool for accepted artifacts; it is not part of the fast search loop.
- Validation metrics are the search signal.
- Test metrics are audit-only and are produced during `accept`.
- The fixed experiment budget for `run` is 300 seconds.
- The saved accepted artifact is retrained on all of `data/train.parquet`.

## Editable Surface

Edit only the local generated `train.py`, specifically:
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
- `feature_importance_source`, `top_feature_importances` when the fitted model exposes native importances
- `n_train_rows`, `n_val_rows`
- `n_features`, `n_base_features`, `n_derived_features`
- `base_feature_names`, `derived_feature_names`, `omitted_base_feature_names`

## Generic Edit Policy

### Core rules

- Make one coherent change per run.
- Prefer reversible, readable edits over large rewrites.
- Search on validation metrics only. Never choose experiments based on test results.
- Use `EXPERIMENT_DESCRIPTION` to state the change type, the concrete edit, and the hypothesis.
- Treat `python run_experiment.py memory-summary` as the source of truth for what has already been tried on the current prepared data.
- Do not rerun the same `candidate_signature` or same `train_sha` unless you are intentionally replicating a result.
- Treat workflow errors, such as an existing artifact directory during `accept`, as operational issues rather than model evidence.

Recommended format:

```python
EXPERIMENT_DESCRIPTION = (
    "move_intent=exploit_current_winner | "
    "change_type=param_refine | family=HistGradientBoosting | "
    "change=max_depth 8->12 | "
    "hypothesis=reduce underfitting on nonlinear interactions"
)
```

Recommended `move_intent` values:
- `explore_new_branch`
  - try a new family, a new feature block, or a genuinely different local branch
- `exploit_current_winner`
  - make a nearby refinement around the current best candidate
- `confirm_replicate`
  - deliberately rerun or closely confirm a result when the gain is small and confidence matters

### Policy Config

Use this machine-readable policy block as the default threshold surface for
external agents and future deterministic controllers:

```yaml
tiny_gain_abs_mape: 0.10
require_confirmation_below_abs_gain: 0.10
max_exploit_streak_before_explore: 3
family_probe_pause_after_clear_losses: 2
branch_cooldown_after_consecutive_losses: 3
regularization_pause_after_losses: 2
```

### Default search order

1. Establish a working baseline if none exists.
2. Probe several model families with mostly stable features.
3. Once one family clearly leads, spend a few runs on local hyperparameter refinement.
4. When model tuning plateaus, try one feature block change.
5. After an accepted feature improvement, retune the current best model family once.

Keep exploration and exploitation balanced:
- after `2-3` consecutive `exploit_current_winner` moves, prefer one `explore_new_branch` move unless the current branch is still improving clearly
- if a move produces only a tiny gain relative to recent deltas, prefer one `confirm_replicate` or one adjacent confirmation move before committing too heavily to that branch

### Confirmation rule

- If a new candidate improves the current best `val_mape` by less than `0.10` absolute MAPE points, do not `accept` it immediately.
- First run either:
  - one `confirm_replicate` of the same candidate, or
  - one adjacent confirmation move that keeps the same branch but tests whether the gain is robust
- Only promote the candidate to `accept` if the confirmation step still supports the improvement.
- If the confirmation step fails, keep the prior accepted candidate even if the first run looked slightly better.

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

For every proposed run, compare the candidate against the nearest prior entry in
`memory-summary` and make sure the change is genuinely novel. The next run should
be clearly different in either:
- model family
- hyperparameter neighborhood
- feature block
- cleanup target

### Switching criteria

- If multiple family probes have not been tried yet, prioritize family exploration before deep tuning.
- If one family is clearly ahead, stay within that family for a short local sweep before switching again.
- If 2 family probes are both clearly worse than the current best family, pause more family probes until the best candidate changes materially or the prepared data changes.
- If 3 consecutive parameter tweaks fail to improve validation metrics, switch to a different family or a feature change.
- If 2 strong regularization moves both shrink the train/validation gap but worsen validation MAPE materially, stop pushing that regularization branch.
- If a feature cleanup wins, prefer either one adjacent cleanup or one local parameter refinement before opening a new model-family branch.
- If a branch has 3 consecutive losses, put that branch on cooldown and do not revisit it immediately.
- If a branch is on cooldown, reopen it only after either:
  - the current best candidate changes materially, or
  - a different exploratory branch also stalls
- If several families are very close, prefer the simpler or faster one.
- If runtime approaches the budget, favor cheaper models or smaller configurations.
- If `train_val_mape_gap` is large or `train_val_mape_ratio` is well above 1.0, simplify or regularize.
- If both train and validation metrics are poor, increase model capacity or improve the feature set.
- If train metrics are strong but validation is weak, prefer regularization, shallower trees, larger leaves, or fewer derived interactions.
- If runtime rises materially without validation gain, prefer the cheaper model or smaller configuration.
- If the derived-feature count grows while validation stalls, try feature cleanup before more model tuning.
- If recent tree-based runs expose `top_feature_importances`, use them as evidence for cleanup and refinement moves rather than pruning derived features blindly.

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
- `same candidate appears repeatedly in memory-summary`
  - stop rerunning it and branch to an adjacent hypothesis instead
- `accept failed because output_dir already exists`
  - rerun `accept` with a unique `--output-dir`; do not treat the failure as a search result

### Search-quality guardrails

- Do not let the search stay in pure exploitation indefinitely.
- Do not let one small win automatically collapse exploration.
- Do not treat every validation gain as equally meaningful; consider whether it is large enough to justify committing the branch.
- Do not revisit a losing branch immediately just because its hypothesis still sounds plausible.
- Prefer search trajectories that make it easy to explain:
  - what branch you are in
  - why the branch is being explored
  - what evidence would cause you to continue, cool down, or abandon it

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
