"""Shared helpers for Braintrust eval parameter handling."""

from __future__ import annotations

from typing import Any

from braintrust.logger import Prompt
from pydantic import BaseModel


def get_hook_parameters(hooks: Any) -> dict[str, Any]:
    """Return raw parameters attached to Braintrust eval hooks."""
    if hooks and hasattr(hooks, "parameters") and isinstance(hooks.parameters, dict):
        return hooks.parameters
    return {}


def unwrap_parameter_value(param: Any, default: Any = None) -> Any:
    """Extract a scalar value from a Braintrust parameter object or class."""
    if param is None:
        return default

    if isinstance(param, BaseModel):
        return getattr(param, "value", param)

    if isinstance(param, type) and issubclass(param, BaseModel):
        try:
            instance = param()
        except Exception:
            return default
        return getattr(instance, "value", instance)

    if hasattr(param, "value"):
        return param.value

    return param


def unwrap_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize all eval parameters into primitive values."""
    unwrapped: dict[str, Any] = {}
    for key, param in params.items():
        value = unwrap_parameter_value(param, default=None)
        if value is not None:
            unwrapped[key] = value
    return unwrapped


def _prompt_content_to_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts)

    return None


def extract_prompt_text_and_model(param: Any) -> tuple[str | None, str | None]:
    """Extract prompt text and embedded model from a Braintrust prompt parameter."""
    if not isinstance(param, Prompt):
        return None, None

    prompt_text: str | None = None
    prompt_block = getattr(param, "prompt", None)
    prompt_type = getattr(prompt_block, "type", None)

    if prompt_type == "completion":
        prompt_text = getattr(prompt_block, "content", None)
    elif prompt_type == "chat":
        messages = getattr(prompt_block, "messages", None) or []
        system_messages = [
            _prompt_content_to_text(getattr(message, "content", None))
            for message in messages
            if getattr(message, "role", None) == "system"
        ]
        prompt_candidates = [text for text in system_messages if text]
        if prompt_candidates:
            prompt_text = "\n\n".join(prompt_candidates)

    options = getattr(param, "options", None) or {}
    model = options.get("model") if isinstance(options, dict) else getattr(options, "model", None)

    return prompt_text, model


def resolve_prompt_and_model(
    params: dict[str, Any],
    *,
    prompt_key: str,
    model_key: str,
    default_model: str,
) -> tuple[str | None, str]:
    """Resolve prompt text and model, preferring prompt-object settings when present."""
    prompt_param = params.get(prompt_key)
    prompt_text, embedded_model = extract_prompt_text_and_model(prompt_param)

    if prompt_text is None:
        prompt_value = unwrap_parameter_value(prompt_param, None)
        if isinstance(prompt_value, str):
            prompt_text = prompt_value

    legacy_model = unwrap_parameter_value(params.get(model_key), default_model)
    resolved_model = embedded_model or legacy_model or default_model
    return prompt_text, str(resolved_model)


def resolve_agent_config_overrides(params: dict[str, Any]) -> dict[str, Any]:
    """Translate eval hook parameters into AgentConfig overrides."""
    overrides = unwrap_parameters(params)

    for prompt_key, model_key in (
        ("system_prompt", "supervisor_model"),
        ("research_agent_prompt", "research_model"),
        ("math_agent_prompt", "math_model"),
    ):
        prompt_text, embedded_model = extract_prompt_text_and_model(params.get(prompt_key))
        if prompt_text is not None:
            overrides[prompt_key] = prompt_text
        if embedded_model is not None and model_key:
            overrides[model_key] = embedded_model

    return overrides
