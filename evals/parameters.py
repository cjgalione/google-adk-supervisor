"""
Parameter definitions for Braintrust evals.

Prompt-bearing parameters use Braintrust's native prompt schema so the
Playground renders prompt/model editors instead of plain text fields.
"""

from typing import Any

from pydantic import BaseModel, Field

from src.config import (
    DEFAULT_MATH_AGENT_PROMPT,
    DEFAULT_MATH_MODEL,
    DEFAULT_RESEARCH_AGENT_PROMPT,
    DEFAULT_RESEARCH_MODEL,
    DEFAULT_SUPERVISOR_MODEL,
    DEFAULT_SYSTEM_PROMPT,
)


def _make_prompt_parameter(
    *,
    default_prompt: str,
    default_model: str,
    description: str,
) -> dict[str, Any]:
    """Build a Braintrust prompt parameter with embedded default model settings."""
    return {
        "type": "prompt",
        "description": description,
        "default": {
            "prompt": {
                "type": "completion",
                "content": default_prompt,
            },
            "options": {
                "model": default_model,
            },
        },
    }


# Define scalar parameters as single-field Pydantic models.
# The patched SDK will extract the 'value' field's schema and default.


class PromptModificationParam(BaseModel):
    """Append-only supervisor prompt modification parameter."""

    value: str = Field(
        default="",
        description=(
            "Optional append-only modification for the supervisor prompt. "
            "Use this to tune routing criteria without replacing the full base prompt."
        ),
    )


SUPERVISOR_EVAL_PARAMETERS = {
    "system_prompt": _make_prompt_parameter(
        default_prompt=DEFAULT_SYSTEM_PROMPT,
        default_model=DEFAULT_SUPERVISOR_MODEL,
        description="Supervisor prompt plus its default model.",
    ),
    "prompt_modification": PromptModificationParam,
    "research_agent_prompt": _make_prompt_parameter(
        default_prompt=DEFAULT_RESEARCH_AGENT_PROMPT,
        default_model=DEFAULT_RESEARCH_MODEL,
        description="Research agent prompt plus its default model.",
    ),
    "math_agent_prompt": _make_prompt_parameter(
        default_prompt=DEFAULT_MATH_AGENT_PROMPT,
        default_model=DEFAULT_MATH_MODEL,
        description="Math agent prompt plus its default model.",
    ),
}

RESEARCH_EVAL_PARAMETERS = {
    "research_agent_prompt": _make_prompt_parameter(
        default_prompt=DEFAULT_RESEARCH_AGENT_PROMPT,
        default_model=DEFAULT_RESEARCH_MODEL,
        description="Research agent prompt plus its default model.",
    ),
}

MATH_EVAL_PARAMETERS = {
    "math_agent_prompt": _make_prompt_parameter(
        default_prompt=DEFAULT_MATH_AGENT_PROMPT,
        default_model=DEFAULT_MATH_MODEL,
        description="Math agent prompt plus its default model.",
    ),
}
