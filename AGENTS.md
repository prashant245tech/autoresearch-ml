# AGENTS.md

This file is the repo-level guide for coding agents working in this project.

## Purpose

This repo is a generic, agent-guided ML experimentation framework:

- `prepare.py` freezes the data boundary
- `train.py` is the editable experiment hypothesis
- `run_experiment.py` is the deterministic evaluation and export harness
- `program.md` is the generic external-agent tuning policy

The design goal is a thin, reliable protocol surface around an editable training spec.

## Core Surfaces

Tracked framework files:

- `prepare.py`
- `run_experiment.py`
- `train.py`
- `program.md`
- `README.md`
- `config/feature_spec.schema.json`
- `tests/`

Local untracked task files:

- `config/feature_spec.json`
- `config/task_context.md`

Generated untracked artifacts:

- `data/`
- `models/accepted/`
- `experiments/session_baseline.json`

Do not move task-specific dataset paths, cohort rules, or business/domain notes into tracked files unless the user explicitly wants a repo example.

## Golden Rules

1. Keep the repo generic.
2. Keep setup state local.
3. Keep evaluation deterministic.
4. Use validation metrics for search decisions.
5. Treat test metrics as audit-only.
6. During normal tuning, edit `train.py` only.

## Setup Workflow

Use this flow when the dataset definition changes:

1. Update `config/task_context.md` if the current task needs local notes.
2. Create or update `config/feature_spec.json`.
3. Validate the spec against `config/feature_spec.schema.json`.
4. Run:

```bash
python prepare.py
```

5. If `experiments/session_baseline.json` exists from an older prep session, delete it before the next tuning run.

Important:

- `filter_specs` exist for columns that need cleaning for cohort filters but should not become model features.
- `train_row_filters` scope the training cohort.
- `test_data_file` and `test_row_filters` support an explicit holdout dataset.

## Tuning Workflow

Use this flow during model iteration:

1. Read `program.md`.
2. Edit `train.py`.
3. Run:

```bash
python run_experiment.py run
```

4. Compare validation metrics against prior accepted runs outside the repo.
5. If accepted, run:

```bash
python run_experiment.py accept --expected-train-sha <train_sha>
```

6. Commit or discard externally.

The repo does not own:

- LLM invocation
- git keep/discard logic
- leaderboard state
- best-model memory

## Editing Guidance

When tuning:

- Prefer one coherent change per run.
- Do not optimize against test metrics.
- Use `EXPERIMENT_DESCRIPTION` to record the change and hypothesis.
- Keep `train.py` readable and reversible.

When changing the framework itself:

- Preserve the thin-harness architecture.
- Do not add dataset-specific logic to tracked files.
- Update docs when the workflow changes.
- Add or update tests for new invariants.

## Testing

Run the framework tests with:

```bash
python3 -m unittest discover -s tests -v
```

The test suite uses temporary workspaces and should not mutate local task files or accepted artifacts.

## Common Pitfalls

- Forgetting to delete `experiments/session_baseline.json` after rerunning `prepare.py`
- Putting real dataset paths into `program.md` or `README.md`
- Using filtered columns without declaring them in `features` or `filter_specs`
- Mixing model-family changes and feature changes in one tuning step
- Treating generated `data/` or model artifacts as source files

## If You Are Unsure

- Ask whether the user wants a framework change or a tuning change.
- Default to preserving the generic framework boundaries.
- If a change affects the prep contract or evaluation protocol, update tests before finishing.
