# AGENTS.md

This file is the repo-level guide for coding agents working in this project.

## Scope

Use this file for repo contract and implementation boundaries.

If you are doing model iteration on a workspace-local generated `train.py`, follow
[program.md](/Users/prashant/Downloads/corrugated_automl%202/program.md) for the
search loop, move taxonomy, confirmation rule, and search-policy thresholds.

## Purpose

This repo is a generic, agent-guided ML experimentation framework:

- `prepare.py` freezes the data boundary inside a workspace
- `baselines/train.generic.py` is the tracked neutral bootstrap baseline
- workspace-local `train.py` is the editable experiment hypothesis
- `run_experiment.py` is the deterministic evaluation and export harness
- `search_memory.py` stores local run/accept history and summaries
- `explain.py` provides optional post-hoc SHAP analysis for accepted artifacts
- `program.md` is the external tuning policy for editing workspace-local `train.py`

The design goal is a thin, reliable protocol surface around an editable local
training spec.

## Core Surfaces

Tracked framework files:

- `prepare.py`
- `run_experiment.py`
- `search_memory.py`
- `explain.py`
- `baselines/train.generic.py`
- `program.md`
- `README.md`
- `config/feature_spec.schema.json`
- `tests/`

Local untracked workspace files:

- `<workspace>/train.py`
- `<workspace>/config/feature_spec.json`
- `<workspace>/config/task_context.md`
- `<workspace>/data/`
- `<workspace>/models/accepted/`
- `<workspace>/experiments/session_baseline.json`
- `<workspace>/experiments/search_memory.jsonl`
- `<workspace>/experiments/search_summary.json`

Do not move real dataset paths, cohort rules, or business/domain notes into
tracked files unless the user explicitly wants a repo example.

## Repo Rules

1. Keep the repo generic.
2. Keep setup state local.
3. Keep evaluation deterministic.
4. Use validation metrics for search decisions.
5. Treat test metrics as audit-only.
6. During normal tuning, edit only the active workspace `train.py`.
7. During framework work, preserve the thin-harness architecture.

## Standard Commands

Setup and prep:

```bash
python prepare.py --workspace <workspace>
```

Bootstrap a local editable train spec:

```bash
python run_experiment.py init-train --workspace <workspace>
```

Run the deterministic evaluator:

```bash
python run_experiment.py run --workspace <workspace>
python run_experiment.py memory-summary --workspace <workspace>
python run_experiment.py accept --workspace <workspace> --expected-train-sha <train_sha>
```

Framework tests:

```bash
python3 -m unittest discover -s tests -v
```

## When Changing The Framework

- Preserve the split:
  - `prepare.py` owns data prep
  - `run_experiment.py` owns protocol
  - `search_memory.py` owns search-memory state and summaries
  - workspace-local `train.py` owns the editable hypothesis
- Do not add dataset-specific logic to tracked framework files.
- Update tests when changing prep validation, runner behavior, search-memory
  semantics, or artifact layout.
- Update docs when changing workflow or file ownership.

## Common Pitfalls

- Forgetting to delete `<workspace>/experiments/session_baseline.json` after rerunning
  `prepare.py` for that workspace
- Putting real dataset paths into tracked docs
- Using filtered columns without declaring them in `features` or `filter_specs`
- Treating generated `data/` or accepted artifacts as source files
- Adding search policy logic to `AGENTS.md` instead of
  [program.md](/Users/prashant/Downloads/corrugated_automl%202/program.md)

## If You Are Unsure

- Ask whether the task is a framework change or a tuning change.
- For tuning changes, read
  [program.md](/Users/prashant/Downloads/corrugated_automl%202/program.md).
- Default to preserving the generic framework boundaries.
