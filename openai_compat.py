import os
import sys
from typing import Optional


_DOTENV_LOADED = False


def load_local_env(env_path: str = ".env") -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED or not os.path.exists(env_path):
        _DOTENV_LOADED = True
        return

    with open(env_path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not key or key in os.environ:
                continue

            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]

            os.environ[key] = value

    _DOTENV_LOADED = True


def resolve_model(explicit_model: Optional[str], env_var: str, default_model: str) -> str:
    load_local_env()
    if explicit_model:
        return explicit_model
    return os.environ.get(env_var) or os.environ.get("OPENAI_MODEL") or default_model


def _extract_message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()

    texts = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                texts.append(part)
                continue

            text = getattr(part, "text", None)
            if text:
                texts.append(text)
                continue

            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                texts.append(part["text"])

    refusal = getattr(message, "refusal", None)
    if refusal and not texts:
        texts.append(refusal)

    return "\n".join(texts).strip()


def chat_completion_text(
    *,
    task_label: str,
    system_prompt: Optional[str],
    user_prompt: str,
    model: str,
    max_tokens: int,
) -> str:
    load_local_env()
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit(f"[{task_label}] openai package not installed. Run: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key and base_url:
        api_key = "dummy"
    if not api_key:
        sys.exit(
            f"[{task_label}] OPENAI_API_KEY not set.\n"
            "         export OPENAI_API_KEY=sk-..."
        )

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )

    if not completion.choices:
        sys.exit(f"[{task_label}] LLM returned no choices.")

    raw = _extract_message_text(completion.choices[0].message)
    if not raw:
        sys.exit(f"[{task_label}] LLM returned an empty response.")
    return raw
