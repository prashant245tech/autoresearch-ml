"""
compile.py — Translates natural language feature descriptions in program.md
             into a structured feature_spec.json using an OpenAI-compatible LLM.

Usage:
    python compile.py                          # Uses OPENAI_API_KEY env var
    python compile.py --preview               # Print spec without saving
    python compile.py --model qwen/qwen3-32b  # Force a specific model id

Flow:
    program.md  →  [OpenAI-compatible API]  →  feature_spec.json  →  prepare.py
"""

import os
import re
import sys
import json
import argparse

from openai_compat import chat_completion_text, resolve_model

PROGRAM_MD    = "program.md"
SPEC_OUT_PATH = "feature_spec.json"

DEFAULT_COMPILE_MODEL = "gpt-4.1-mini"

# ─────────────────────────────────────────────────────────────
# Parse sections from program.md
# ─────────────────────────────────────────────────────────────

def parse_config_block(content: str) -> dict:
    """Extract the ```config ... ``` block and parse as simple key:value."""
    match = re.search(r"```config\s*\n(.*?)```", content, re.DOTALL)
    if not match:
        sys.exit("[compile] ERROR: No ```config ... ``` block found in program.md")
    cfg = {}
    for line in match.group(1).strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            cfg[k.strip()] = v.strip()
    required = ["data_file", "target_column"]
    for r in required:
        if r not in cfg:
            sys.exit(f"[compile] ERROR: '{r}' not found in config block")
    return cfg


def parse_features_block(content: str) -> str:
    """Extract the ```features ... ``` block as raw text."""
    match = re.search(r"```features\s*\n(.*?)```", content, re.DOTALL)
    if not match:
        sys.exit("[compile] ERROR: No ```features ... ``` block found in program.md")
    return match.group(1).strip()


def parse_domain_context(content: str) -> str:
    """Extract Section 3 domain context for embedding in spec."""
    match = re.search(
        r"SECTION 3.*?────+\s*\n(.*?)(?=\Z|##\s*──)",
        content, re.DOTALL
    )
    return match.group(1).strip() if match else ""


# ─────────────────────────────────────────────────────────────
# LLM prompt
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a data engineering assistant. Your job is to translate
natural language feature descriptions into a precise JSON specification
that a generic data preparation pipeline can execute deterministically.

You will receive:
1. A list of feature descriptions written in plain English
2. The target column name

You must return a single valid JSON object — no markdown, no explanation, no code fences.
The JSON must strictly follow the schema below.

SCHEMA:
{
  "features": [
    {
      "column": "<exact CSV column name>",
      "type": "<numeric|ordinal|categorical|boolean>",
      "required": <true|false>,
      "drop_if_null": <true|false>,
      "fill_null": {
        "strategy": "<value|median|mean|mode|none>",
        "value": <only present when strategy is "value">
      },
      "clip": {
        "min": <number or null>,
        "max": <number or null>
      },
      "normalise": "<none|uppercase_strip>",
      "ordinal_order": ["<val1>", "<val2>", ...],  // ordinal only, low→high
      "categorical_consolidation": {               // categorical only, optional
        "<raw_value>": "<canonical_value>",
        ...
      },
      "unknown_sentinel": <-1 for ordinal, "UNKNOWN" for categorical, 0 for boolean>,
      "notes": "<brief reason for these choices>"
    }
  ]
}

RULES:
- type must be one of: numeric, ordinal, categorical, boolean
- ordinal: encode as integer rank (1 = lowest). Unknown → unknown_sentinel (-1).
- categorical: label-encode after applying normalise and consolidation. Unknown → "UNKNOWN".
- boolean: encode as 0/1 integer.
- numeric: keep as float after clip and fill.
- drop_if_null true means the entire row is dropped if this column is null.
- fill_null.strategy "none" means leave nulls as NaN (for models that handle it).
- column names must match the descriptions EXACTLY — preserve spaces, slashes, % signs.
- ordinal_order values should be UPPERCASE and normalised (no spaces).
- categorical_consolidation maps messy raw values to canonical values BEFORE label encoding.
- Keep notes concise — one sentence max.
- Return ONLY the JSON. No preamble, no explanation, no markdown.
"""


def build_user_prompt(features_text: str, target_col: str) -> str:
    return f"""Target column: {target_col}

Feature descriptions:
{features_text}

Translate each feature into the JSON spec. Return only the JSON object."""


# ─────────────────────────────────────────────────────────────
# Call OpenAI-compatible API
# ─────────────────────────────────────────────────────────────

def call_llm(features_text: str, target_col: str, model: str) -> dict:
    print(f"[compile] Calling model `{model}` to codify feature descriptions...")

    raw = chat_completion_text(
        task_label="compile",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(features_text, target_col),
        model=model,
        max_tokens=4096,
    )

    # Strip accidental markdown code fences
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[compile] ERROR: LLM returned invalid JSON:\n{raw[:500]}")
        sys.exit(f"JSON parse error: {e}")

    if "features" not in spec:
        sys.exit("[compile] ERROR: LLM response missing top-level 'features' key")

    return spec


# ─────────────────────────────────────────────────────────────
# Merge config + spec and save
# ─────────────────────────────────────────────────────────────

def build_full_spec(cfg: dict, feature_spec: dict, domain_context: str) -> dict:
    return {
        "version":        "1.0",
        "data_file":      cfg["data_file"],
        "target_column":  cfg["target_column"],
        "test_size":      float(cfg.get("test_size", 0.20)),
        "random_state":   int(cfg.get("random_state", 42)),
        "outlier_iqr_k":  float(cfg.get("outlier_iqr_k", 4.0)),
        "domain_context": domain_context,
        "features":       feature_spec["features"],
    }


def print_spec_summary(spec: dict):
    print(f"\n{'═'*60}")
    print(f"  FEATURE SPEC SUMMARY")
    print(f"{'═'*60}")
    print(f"  Data file   : {spec['data_file']}")
    print(f"  Target      : {spec['target_column']}")
    print(f"  Test size   : {spec['test_size']}")
    print(f"  Features    : {len(spec['features'])}")
    print(f"{'─'*60}")
    for f in spec["features"]:
        req_marker = " *" if f.get("drop_if_null") else "  "
        fill_str   = ""
        fn = f.get("fill_null", {})
        if fn.get("strategy") == "value":
            fill_str = f"fill={fn.get('value')}"
        elif fn.get("strategy") not in (None, "none"):
            fill_str = f"fill={fn['strategy']}"
        clip = f.get("clip", {})
        clip_str = ""
        if clip.get("min") is not None or clip.get("max") is not None:
            clip_str = f"clip[{clip.get('min','')},{clip.get('max','')}]"
        extras = " | ".join(filter(None, [fill_str, clip_str]))
        print(f"{req_marker} {f['column']:30s}  {f['type']:12s}  {extras}")
    print(f"{'─'*60}")
    print(f"  (* = required, row dropped if null)")
    print(f"{'═'*60}\n")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compile program.md feature descriptions into feature_spec.json"
    )
    parser.add_argument("--preview", action="store_true",
                        help="Print spec to stdout without saving")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "OpenAI-compatible model id to use "
            f"(default: $COMPILE_LLM_MODEL, then $OPENAI_MODEL, then {DEFAULT_COMPILE_MODEL})"
        ),
    )
    args = parser.parse_args()

    # ── Read program.md ───────────────────────────────────────
    if not os.path.exists(PROGRAM_MD):
        sys.exit(f"[compile] ERROR: {PROGRAM_MD} not found")

    with open(PROGRAM_MD) as f:
        content = f.read()

    cfg            = parse_config_block(content)
    features_text  = parse_features_block(content)
    domain_context = parse_domain_context(content)

    print(f"[compile] Parsed {PROGRAM_MD}")
    print(f"[compile] Target   : {cfg['target_column']}")
    print(f"[compile] Data file: {cfg['data_file']}")
    feature_names = [
        line.split(":")[0].strip()
        for line in features_text.splitlines()
        if line.strip() and not line.startswith(" ") and ":" in line
    ]
    print(f"[compile] Features described: {feature_names}")

    # ── Call LLM ──────────────────────────────────────────────
    model = resolve_model(args.model, "COMPILE_LLM_MODEL", DEFAULT_COMPILE_MODEL)
    feature_spec   = call_llm(features_text, cfg["target_column"], model)
    full_spec      = build_full_spec(cfg, feature_spec, domain_context)

    # ── Output ────────────────────────────────────────────────
    print_spec_summary(full_spec)

    if args.preview:
        print(json.dumps(full_spec, indent=2))
        print("\n[compile] Preview only — feature_spec.json NOT saved (--preview mode)")
    else:
        with open(SPEC_OUT_PATH, "w") as f:
            json.dump(full_spec, f, indent=2)
        print(f"[compile] ✅ Saved → {SPEC_OUT_PATH}")
        print(f"[compile] Review the spec, then run: python prepare.py")


if __name__ == "__main__":
    main()
