"""
search_memory.py - Local search-memory storage and summary helpers.

This module owns:
- append-only JSONL search-memory events
- stable candidate signatures
- prepared-data-scoped summary generation
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Optional
from uuid import uuid4

SEARCH_MEMORY_PATH = "experiments/search_memory.jsonl"
SEARCH_SUMMARY_PATH = "experiments/search_summary.json"
SUMMARY_FORMAT_VERSION = 5

EVENT_TYPE_RUN = "run"
EVENT_TYPE_ACCEPT = "accept"
STATUS_OK = "ok"

SUMMARY_RECENT_EVENTS_LIMIT = 10
SUMMARY_TOP_CANDIDATES_LIMIT = 10

NON_SEARCH_RELEVANT_STATUSES = {"prepared_data_mismatch", "hash_mismatch"}

DESCRIPTION_UNKNOWN = "unknown"


def _json_stable(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def parse_experiment_description(description: Optional[str]) -> dict:
    parsed = {
        "move_intent": DESCRIPTION_UNKNOWN,
        "change_type": DESCRIPTION_UNKNOWN,
        "declared_family": DESCRIPTION_UNKNOWN,
        "change_summary": None,
        "hypothesis": None,
    }
    if not description or not isinstance(description, str):
        return parsed

    fields = {}
    for raw_part in description.split("|"):
        part = raw_part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            fields[key] = value

    parsed["move_intent"] = fields.get("move_intent", DESCRIPTION_UNKNOWN) or DESCRIPTION_UNKNOWN
    parsed["change_type"] = fields.get("change_type", DESCRIPTION_UNKNOWN) or DESCRIPTION_UNKNOWN
    parsed["declared_family"] = fields.get("family", DESCRIPTION_UNKNOWN) or DESCRIPTION_UNKNOWN
    parsed["change_summary"] = fields.get("change")
    parsed["hypothesis"] = fields.get("hypothesis")
    return parsed


def description_fields_for_event(event: dict) -> dict:
    parsed = parse_experiment_description(event.get("experiment_description"))
    move_intent = event.get("move_intent")
    change_type = event.get("change_type")
    declared_family = event.get("declared_family")
    change_summary = event.get("change_summary")
    hypothesis = event.get("hypothesis")
    if move_intent:
        parsed["move_intent"] = move_intent
    if change_type:
        parsed["change_type"] = change_type
    if declared_family:
        parsed["declared_family"] = declared_family
    if change_summary:
        parsed["change_summary"] = change_summary
    if hypothesis:
        parsed["hypothesis"] = hypothesis
    return parsed


def build_model_signature(model_class: Optional[str], model_params: Optional[dict]) -> Optional[str]:
    if not model_class:
        return None
    payload = {
        "model_class": model_class,
        "model_params": model_params or {},
    }
    return _sha256_text(_json_stable(payload))


def build_feature_signature(feature_names: Optional[list[str]]) -> Optional[str]:
    if not feature_names:
        return None
    return _sha256_text(_json_stable(list(feature_names)))


def build_candidate_signature(
    prepared_data_sha: Optional[str],
    model_signature: Optional[str],
    feature_signature: Optional[str],
) -> Optional[str]:
    if not prepared_data_sha or not model_signature or not feature_signature:
        return None
    payload = {
        "prepared_data_sha": prepared_data_sha,
        "model_signature": model_signature,
        "feature_signature": feature_signature,
    }
    return _sha256_text(_json_stable(payload))


def is_search_relevant_event(event_type: str, status: Optional[str]) -> bool:
    if not status or status in NON_SEARCH_RELEVANT_STATUSES:
        return False
    # Accept failures are workflow noise, not useful search evidence.
    if event_type == EVENT_TYPE_ACCEPT and status != STATUS_OK:
        return False
    return True


def _simplified_recent_event(event: dict) -> dict:
    description_fields = description_fields_for_event(event)
    payload = {
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "recorded_at": event.get("recorded_at"),
        "status": event.get("status"),
        "train_sha": event.get("train_sha"),
        "candidate_signature": event.get("candidate_signature"),
        "model_class": event.get("model_class"),
        "experiment_description": event.get("experiment_description"),
        "move_intent": description_fields["move_intent"],
        "change_type": description_fields["change_type"],
        "declared_family": description_fields["declared_family"],
    }
    if event.get("event_type") == EVENT_TYPE_RUN:
        payload["val_mape"] = event.get("val_mape")
        payload["train_mape"] = event.get("train_mape")
        payload["feature_importance_source"] = event.get("feature_importance_source")
        payload["top_feature_importances"] = event.get("top_feature_importances")
    if event.get("event_type") == EVENT_TYPE_ACCEPT:
        payload["test_mape"] = event.get("test_mape")
        payload["output_dir"] = event.get("output_dir")
    if event.get("error"):
        payload["error"] = event.get("error")
    return payload


def _build_empty_summary(prepared_data_sha: Optional[str]) -> dict:
    return {
        "status": "ok",
        "summary_format_version": SUMMARY_FORMAT_VERSION,
        "prepared_data_sha": prepared_data_sha,
        "search_memory_path": SEARCH_MEMORY_PATH,
        "search_summary_path": SEARCH_SUMMARY_PATH,
        "counts": {
            "total_events": 0,
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "accepts": 0,
            "failed_accepts": 0,
        },
        "best_run": None,
        "top_unique_candidates": [],
        "recent_events": [],
        "family_stats": {},
        "duplicate_candidate_counts": {},
        "accepted_candidates": [],
        "repeated_exact_runs": [],
        "move_intent_distribution": {},
        "family_branch_depth": None,
    }


def build_event(event_type: str, payload: dict) -> Optional[dict]:
    status = payload.get("status")
    if not is_search_relevant_event(event_type, status):
        return None

    model_signature = build_model_signature(
        payload.get("model_class"),
        payload.get("model_params"),
    )
    feature_signature = build_feature_signature(payload.get("feature_names"))
    candidate_signature = build_candidate_signature(
        payload.get("prepared_data_sha"),
        model_signature,
        feature_signature,
    )
    parsed_description = parse_experiment_description(payload.get("experiment_description"))

    event = {
        "event_id": uuid4().hex,
        "event_type": event_type,
        "recorded_at": _timestamp(),
        "prepared_data_sha": payload.get("prepared_data_sha"),
        "train_sha": payload.get("train_sha"),
        "experiment_description": payload.get("experiment_description"),
        "status": status,
        "model_signature": model_signature,
        "feature_signature": feature_signature,
        "candidate_signature": candidate_signature,
        **parsed_description,
    }

    if payload.get("error"):
        event["error"] = payload.get("error")

    if event_type == EVENT_TYPE_RUN:
        for key in [
            "budget_s",
            "elapsed_s",
            "elapsed_budget_fraction",
            "model_class",
            "model_params",
            "feature_importance_source",
            "top_feature_importances",
            "feature_names",
            "n_features",
            "n_train_rows",
            "n_val_rows",
            "n_base_features",
            "n_derived_features",
            "n_candidate_base_features",
            "base_feature_names",
            "derived_feature_names",
            "omitted_base_feature_names",
            "fit_elapsed_s",
            "predict_train_elapsed_s",
            "predict_val_elapsed_s",
            "train_mape",
            "train_rmse",
            "train_r2",
            "val_mape",
            "val_rmse",
            "val_r2",
            "train_val_mape_gap",
            "train_val_rmse_gap",
            "train_val_mape_ratio",
        ]:
            if key in payload:
                event[key] = payload.get(key)

    if event_type == EVENT_TYPE_ACCEPT:
        for key in [
            "elapsed_s",
            "fit_elapsed_s",
            "predict_test_elapsed_s",
            "model_class",
            "model_params",
            "feature_names",
            "n_features",
            "n_train_rows",
            "n_test_rows",
            "n_base_features",
            "n_derived_features",
            "n_candidate_base_features",
            "base_feature_names",
            "derived_feature_names",
            "omitted_base_feature_names",
            "test_mape",
            "test_rmse",
            "test_r2",
            "output_dir",
        ]:
            if key in payload:
                event[key] = payload.get(key)

    return event


def append_event(event: dict, path: str = SEARCH_MEMORY_PATH) -> None:
    _ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(_json_stable(event))
        handle.write("\n")


def load_events(path: str = SEARCH_MEMORY_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []

    events = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                events.append(json.loads(raw_line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path} at line {line_no}: {exc}"
                ) from exc
    return events


def _resolve_scope_prepared_data_sha(
    events: list[dict], preferred_prepared_data_sha: Optional[str]
) -> Optional[str]:
    if preferred_prepared_data_sha:
        return preferred_prepared_data_sha
    if not events:
        return None
    return events[-1].get("prepared_data_sha")


def _best_run_payload(event: dict) -> dict:
    description_fields = description_fields_for_event(event)
    return {
        "candidate_signature": event.get("candidate_signature"),
        "feature_signature": event.get("feature_signature"),
        "model_signature": event.get("model_signature"),
        "train_sha": event.get("train_sha"),
        "model_class": event.get("model_class"),
        "experiment_description": event.get("experiment_description"),
        "move_intent": description_fields["move_intent"],
        "change_type": description_fields["change_type"],
        "declared_family": description_fields["declared_family"],
        "feature_importance_source": event.get("feature_importance_source"),
        "top_feature_importances": event.get("top_feature_importances"),
        "val_mape": event.get("val_mape"),
        "val_rmse": event.get("val_rmse"),
        "val_r2": event.get("val_r2"),
        "train_mape": event.get("train_mape"),
        "n_features": event.get("n_features"),
        "n_base_features": event.get("n_base_features"),
        "n_derived_features": event.get("n_derived_features"),
        "recorded_at": event.get("recorded_at"),
    }


def _select_best_attempt(events: list[dict], metric_key: str) -> dict:
    best_metric = min(event.get(metric_key, float("inf")) for event in events)
    best_events = [
        event
        for event in events
        if event.get(metric_key, float("inf")) == best_metric
    ]
    return max(best_events, key=lambda event: event.get("recorded_at") or "")


def _build_move_intent_distribution(run_events: list[dict]) -> dict:
    counts = {}
    for event in run_events:
        move_intent = description_fields_for_event(event)["move_intent"]
        counts[move_intent] = counts.get(move_intent, 0) + 1
    return dict(sorted(counts.items()))


def _build_family_branch_depth(successful_runs: list[dict]) -> Optional[dict]:
    if not successful_runs:
        return None

    ordered_runs = sorted(successful_runs, key=lambda event: event.get("recorded_at") or "")
    latest = ordered_runs[-1]
    current_model_class = latest.get("model_class")
    if not current_model_class:
        return None
    latest_description_fields = description_fields_for_event(latest)

    depth = 0
    branch_start_recorded_at = latest.get("recorded_at")
    branch_start_train_sha = latest.get("train_sha")
    branch_move_intents = []

    for event in reversed(ordered_runs):
        if event.get("model_class") != current_model_class:
            break
        description_fields = description_fields_for_event(event)
        depth += 1
        branch_start_recorded_at = event.get("recorded_at")
        branch_start_train_sha = event.get("train_sha")
        branch_move_intents.append(description_fields["move_intent"])
        if description_fields["move_intent"] == "explore_new_branch":
            break

    branch_move_intents.reverse()
    return {
        "model_class": current_model_class,
        "declared_family": latest_description_fields["declared_family"],
        "depth": depth,
        "latest_move_intent": latest_description_fields["move_intent"],
        "branch_start_recorded_at": branch_start_recorded_at,
        "branch_start_train_sha": branch_start_train_sha,
        "branch_move_intents": branch_move_intents,
    }


def build_summary(
    events: list[dict],
    prepared_data_sha: Optional[str],
) -> dict:
    scope_sha = _resolve_scope_prepared_data_sha(events, prepared_data_sha)
    scoped_events = [event for event in events if event.get("prepared_data_sha") == scope_sha]

    summary = _build_empty_summary(scope_sha)
    if not scoped_events:
        return summary

    run_events = [event for event in scoped_events if event.get("event_type") == EVENT_TYPE_RUN]
    successful_runs = [event for event in run_events if event.get("status") == STATUS_OK]
    failed_runs = [event for event in run_events if event.get("status") != STATUS_OK]
    accept_events = [event for event in scoped_events if event.get("event_type") == EVENT_TYPE_ACCEPT]
    successful_accepts = [event for event in accept_events if event.get("status") == STATUS_OK]
    failed_accepts = [event for event in accept_events if event.get("status") != STATUS_OK]

    summary["counts"] = {
        "total_events": len(scoped_events),
        "total_runs": len(run_events),
        "successful_runs": len(successful_runs),
        "failed_runs": len(failed_runs),
        "accepts": len(successful_accepts),
        "failed_accepts": len(failed_accepts),
    }
    summary["move_intent_distribution"] = _build_move_intent_distribution(run_events)
    summary["family_branch_depth"] = _build_family_branch_depth(successful_runs)

    if successful_runs:
        best_run = _select_best_attempt(successful_runs, "val_mape")
        summary["best_run"] = _best_run_payload(best_run)

    candidate_groups = {}
    duplicate_candidate_counts = {}
    repeated_exact_groups = {}
    family_stats = {}

    for event in successful_runs:
        candidate_signature = event.get("candidate_signature")
        if candidate_signature:
            candidate_groups.setdefault(candidate_signature, []).append(event)
            duplicate_candidate_counts[candidate_signature] = (
                duplicate_candidate_counts.get(candidate_signature, 0) + 1
            )

            exact_key = (event.get("train_sha"), candidate_signature)
            repeated_exact_groups.setdefault(exact_key, []).append(event)

        model_class = event.get("model_class")
        if model_class:
            stats = family_stats.setdefault(
                model_class,
                {
                    "successful_run_count": 0,
                    "best_val_mape": None,
                    "avg_val_mape": None,
                    "accept_count": 0,
                    "latest_train_sha": None,
                },
            )
            stats["successful_run_count"] += 1
            val_mape = event.get("val_mape")
            if isinstance(val_mape, (int, float)):
                if stats["best_val_mape"] is None or val_mape < stats["best_val_mape"]:
                    stats["best_val_mape"] = val_mape
            stats["latest_train_sha"] = event.get("train_sha")

    top_unique_candidates = []
    for candidate_signature, attempts in candidate_groups.items():
        best_attempt = _select_best_attempt(attempts, "val_mape")
        description_fields = description_fields_for_event(best_attempt)
        accept_count = sum(
            1
            for event in successful_accepts
            if event.get("candidate_signature") == candidate_signature
        )
        top_unique_candidates.append(
            {
                "candidate_signature": candidate_signature,
                "model_signature": best_attempt.get("model_signature"),
                "feature_signature": best_attempt.get("feature_signature"),
                "train_sha": best_attempt.get("train_sha"),
                "model_class": best_attempt.get("model_class"),
                "experiment_description": best_attempt.get("experiment_description"),
                "move_intent": description_fields["move_intent"],
                "change_type": description_fields["change_type"],
                "declared_family": description_fields["declared_family"],
                "feature_importance_source": best_attempt.get("feature_importance_source"),
                "top_feature_importances": best_attempt.get("top_feature_importances"),
                "best_val_mape": best_attempt.get("val_mape"),
                "best_val_rmse": best_attempt.get("val_rmse"),
                "best_val_r2": best_attempt.get("val_r2"),
                "attempt_count": len(attempts),
                "accept_count": accept_count,
                "n_features": best_attempt.get("n_features"),
                "n_base_features": best_attempt.get("n_base_features"),
                "n_derived_features": best_attempt.get("n_derived_features"),
            }
        )

    top_unique_candidates.sort(
        key=lambda item: (
            item.get("best_val_mape", float("inf")),
            item.get("experiment_description") or "",
        )
    )
    summary["top_unique_candidates"] = top_unique_candidates[:SUMMARY_TOP_CANDIDATES_LIMIT]

    summary["duplicate_candidate_counts"] = {
        candidate_signature: count
        for candidate_signature, count in sorted(duplicate_candidate_counts.items())
        if count > 1
    }

    repeated_exact_runs = []
    for (train_sha, candidate_signature), attempts in repeated_exact_groups.items():
        if len(attempts) <= 1:
            continue
        latest_attempt = attempts[-1]
        repeated_exact_runs.append(
            {
                "train_sha": train_sha,
                "candidate_signature": candidate_signature,
                "attempt_count": len(attempts),
                "latest_recorded_at": latest_attempt.get("recorded_at"),
                "val_mape": latest_attempt.get("val_mape"),
            }
        )
    repeated_exact_runs.sort(key=lambda item: (-item["attempt_count"], item["train_sha"] or ""))
    summary["repeated_exact_runs"] = repeated_exact_runs

    for model_class, stats in family_stats.items():
        class_events = [event for event in successful_runs if event.get("model_class") == model_class]
        class_val_mapes = [
            event.get("val_mape")
            for event in class_events
            if isinstance(event.get("val_mape"), (int, float))
        ]
        if class_val_mapes:
            stats["avg_val_mape"] = round(sum(class_val_mapes) / len(class_val_mapes), 4)
        stats["accept_count"] = sum(
            1 for event in successful_accepts if event.get("model_class") == model_class
        )
    summary["family_stats"] = family_stats

    accepted_candidates = {}
    for event in successful_accepts:
        description_fields = description_fields_for_event(event)
        candidate_signature = event.get("candidate_signature")
        key = candidate_signature or event.get("event_id")
        existing = accepted_candidates.get(key)
        accept_count = 1 if existing is None else existing["accept_count"] + 1
        accepted_candidates[key] = {
            "candidate_signature": candidate_signature,
            "train_sha": event.get("train_sha"),
            "model_class": event.get("model_class"),
            "experiment_description": event.get("experiment_description"),
            "move_intent": description_fields["move_intent"],
            "change_type": description_fields["change_type"],
            "declared_family": description_fields["declared_family"],
            "test_mape": event.get("test_mape"),
            "test_rmse": event.get("test_rmse"),
            "test_r2": event.get("test_r2"),
            "output_dir": event.get("output_dir"),
            "accept_count": accept_count,
            "latest_recorded_at": event.get("recorded_at"),
        }
    accepted_candidate_summaries = list(accepted_candidates.values())
    accepted_candidate_summaries.sort(
        key=lambda item: item.get("latest_recorded_at") or "",
        reverse=True,
    )
    summary["accepted_candidates"] = accepted_candidate_summaries

    summary["recent_events"] = [
        _simplified_recent_event(event)
        for event in scoped_events[-SUMMARY_RECENT_EVENTS_LIMIT:]
    ]

    return summary


def write_summary(summary: dict, path: str = SEARCH_SUMMARY_PATH) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def rebuild_summary(
    prepared_data_sha: Optional[str] = None,
    events_path: str = SEARCH_MEMORY_PATH,
    summary_path: str = SEARCH_SUMMARY_PATH,
) -> dict:
    events = load_events(events_path)
    summary = build_summary(events, prepared_data_sha)
    write_summary(summary, summary_path)
    return summary


def load_summary(path: str = SEARCH_SUMMARY_PATH) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def get_or_rebuild_summary(
    prepared_data_sha: Optional[str] = None,
    events_path: str = SEARCH_MEMORY_PATH,
    summary_path: str = SEARCH_SUMMARY_PATH,
) -> dict:
    existing = load_summary(summary_path)
    if (
        existing is not None
        and existing.get("prepared_data_sha") == prepared_data_sha
        and existing.get("summary_format_version") == SUMMARY_FORMAT_VERSION
    ):
        return existing
    return rebuild_summary(prepared_data_sha, events_path=events_path, summary_path=summary_path)


def record_event(
    event_type: str,
    payload: dict,
    events_path: str = SEARCH_MEMORY_PATH,
    summary_path: str = SEARCH_SUMMARY_PATH,
) -> Optional[dict]:
    event = build_event(event_type, payload)
    if event is None:
        return None
    append_event(event, path=events_path)
    rebuild_summary(
        prepared_data_sha=event.get("prepared_data_sha"),
        events_path=events_path,
        summary_path=summary_path,
    )
    return event
