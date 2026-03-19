from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic_settings import BaseSettings

CONFIG_FILE = Path(__file__).parent / "config.json"


EDITABLE_FIELDS = frozenset({
    "root_model",
    "rlm_threshold_chars", "max_iterations", "max_sub_calls",
    "sub_call_max_chars", "repl_exec_timeout",
    "passthrough_short", "token_estimate_ratio",
    "sub_call_cache_size", "max_concurrent_sub_calls",
    "api_key", "metrics_enabled", "prompt_profile_override",
    "trajectory_log_dir", "max_concurrent_sessions", "max_queue_size",
    "history_compact_threshold", "ollama_max_retries", "ollama_retry_base_delay",
    "max_sub_call_depth", "token_budget",
    "circuit_breaker_threshold", "circuit_breaker_timeout",
    "ollama_hosts", "dispatcher_refresh_interval",
})


class Settings(BaseSettings):
    # Ollama connection
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_hosts: str = os.getenv("RLM_OLLAMA_HOSTS", "")  # comma-separated, empty = use ollama_base_url
    dispatcher_refresh_interval: int = int(os.getenv("RLM_DISPATCHER_REFRESH_INTERVAL", "60"))

    # Model selection (root model for RLM code generation; sub model = user's requested model)
    # Root model for RLM code generation. Use "provider/model" to pin to a provider.
    root_model: str = os.getenv("RLM_ROOT_MODEL", "qwen3-coder-next")

    # RLM behaviour
    rlm_threshold_chars: int = int(os.getenv("RLM_THRESHOLD_CHARS", "50000"))
    max_iterations: int = int(os.getenv("RLM_MAX_ITERATIONS", "20"))
    max_sub_calls: int = int(os.getenv("RLM_MAX_SUB_CALLS", "50"))
    sub_call_max_chars: int = int(os.getenv("RLM_SUB_CALL_MAX_CHARS", "500000"))
    repl_exec_timeout: int = int(os.getenv("RLM_EXEC_TIMEOUT", "60"))

    # Server
    host: str = os.getenv("RLM_HOST", "0.0.0.0")
    port: int = int(os.getenv("RLM_PORT", "8881"))

    # Passthrough: forward short prompts directly to Ollama
    passthrough_short: bool = os.getenv("RLM_PASSTHROUGH_SHORT", "true").lower() == "true"

    # Token estimation
    token_estimate_ratio: float = float(os.getenv("RLM_TOKEN_ESTIMATE_RATIO", "4.0"))

    # Sub-call caching
    sub_call_cache_size: int = int(os.getenv("RLM_SUB_CALL_CACHE_SIZE", "128"))

    # Parallel sub-calls
    max_concurrent_sub_calls: int = int(os.getenv("RLM_MAX_CONCURRENT_SUB_CALLS", "4"))

    # Authentication
    api_key: str = os.getenv("RLM_API_KEY", "")

    # Observability
    metrics_enabled: bool = os.getenv("RLM_METRICS_ENABLED", "true").lower() == "true"

    # Prompt profiles
    prompt_profile_override: str = os.getenv("RLM_PROMPT_PROFILE", "")

    # Trajectory logging
    trajectory_log_dir: str = os.getenv("RLM_TRAJECTORY_LOG_DIR", "")

    # Request queue
    max_concurrent_sessions: int = int(os.getenv("RLM_MAX_CONCURRENT_SESSIONS", "2"))
    max_queue_size: int = int(os.getenv("RLM_MAX_QUEUE_SIZE", "10"))

    # History compaction
    history_compact_threshold: int = int(os.getenv("RLM_HISTORY_COMPACT_THRESHOLD", "30"))

    # Retry
    ollama_max_retries: int = int(os.getenv("RLM_OLLAMA_MAX_RETRIES", "3"))
    ollama_retry_base_delay: float = float(os.getenv("RLM_OLLAMA_RETRY_BASE_DELAY", "1.0"))

    # Sub-call depth limit
    max_sub_call_depth: int = int(os.getenv("RLM_MAX_SUB_CALL_DEPTH", "3"))

    # Token budget (0 = unlimited)
    token_budget: int = int(os.getenv("RLM_TOKEN_BUDGET", "0"))

    # Circuit breaker
    circuit_breaker_threshold: int = int(os.getenv("RLM_CIRCUIT_BREAKER_THRESHOLD", "5"))
    circuit_breaker_timeout: int = int(os.getenv("RLM_CIRCUIT_BREAKER_TIMEOUT", "30"))

    model_config = {"env_prefix": "RLM_", "extra": "ignore"}


settings = Settings()


def get_all_settings() -> dict:
    """Return all current settings as a JSON-serializable dict."""
    return {k: v for k, v in settings.model_dump().items() if k != "model_config"}


def update_settings(updates: dict) -> dict[str, str]:
    """Update settings in memory. Returns dict of errors for invalid keys."""
    errors = {}
    for key, value in updates.items():
        if key not in EDITABLE_FIELDS:
            errors[key] = f"Field '{key}' is not editable"
            continue
        if not hasattr(settings, key):
            errors[key] = f"Unknown field '{key}'"
            continue
        try:
            # Type coerce based on current type
            current = getattr(settings, key)
            if isinstance(current, bool):
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes")
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
            object.__setattr__(settings, key, value)
        except (ValueError, TypeError) as exc:
            errors[key] = str(exc)
    return errors


def save_settings(providers: list[dict] | None = None) -> str:
    """Save current editable settings to config.json. Returns the file path."""
    data = {k: getattr(settings, k) for k in EDITABLE_FIELDS if hasattr(settings, k)}
    if providers is not None:
        data["providers"] = providers
    CONFIG_FILE.write_text(json.dumps(data, indent=2, default=str) + "\n")
    return str(CONFIG_FILE)


def load_settings() -> None:
    """Reload settings from env vars, then overlay config.json if it exists."""
    fresh = Settings()
    # Copy all fields from fresh instance
    for key in Settings.model_fields:
        object.__setattr__(settings, key, getattr(fresh, key))
    # Overlay config.json
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text())
            update_settings(data)
        except Exception:
            pass
