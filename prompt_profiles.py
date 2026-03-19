"""Model-specific prompt customization profiles."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PromptProfile:
    """Customization for how the RLM system prompt is presented to different models."""
    name: str
    # Extra instructions appended to the system prompt
    extra_instructions: str = ""
    # Override temperature if set (None = use caller's default)
    temperature_override: float | None = None
    # Preferred code style hints
    code_style_hint: str = ""
    # Max sub-call batch size suggestion
    suggested_batch_size: int | None = None


PROFILES: dict[str, PromptProfile] = {
    "default": PromptProfile(
        name="default",
    ),
    "qwen3-coder": PromptProfile(
        name="qwen3-coder",
        extra_instructions=(
            "You excel at Python code generation. Use list comprehensions and "
            "built-in functions where appropriate. Be concise in your code."
        ),
        code_style_hint="pythonic, concise",
        suggested_batch_size=5,
    ),
    "qwen3.5": PromptProfile(
        name="qwen3.5",
        extra_instructions=(
            "You have a large context window. Consider processing larger chunks "
            "in each sub-call to minimize the total number of calls."
        ),
        suggested_batch_size=3,
    ),
    "llama": PromptProfile(
        name="llama",
        extra_instructions=(
            "Keep your code blocks simple and straightforward. "
            "Prefer explicit loops over complex comprehensions."
        ),
        code_style_hint="explicit, step-by-step",
        suggested_batch_size=4,
    ),
    "deepseek-coder": PromptProfile(
        name="deepseek-coder",
        extra_instructions=(
            "You are skilled at code analysis. Use efficient string operations "
            "and regular expressions when processing text."
        ),
        code_style_hint="efficient, regex-friendly",
        suggested_batch_size=5,
    ),
}


def get_profile(model_name: str) -> PromptProfile:
    """Get the prompt profile for a model, using substring matching.

    Falls back to 'default' if no match found.
    Checks settings.prompt_profile_override first.
    """
    from config import settings

    # Check for explicit override
    if settings.prompt_profile_override:
        override = settings.prompt_profile_override.lower()
        if override in PROFILES:
            return PROFILES[override]

    # Substring match against model name
    model_lower = model_name.lower()
    for key, profile in PROFILES.items():
        if key != "default" and key in model_lower:
            return profile

    return PROFILES["default"]
