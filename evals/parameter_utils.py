"""Shared helpers for Braintrust eval parameter handling."""

from __future__ import annotations

from typing import Any

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

    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            text = (
                part.get("text")
                if isinstance(part, dict)
                else getattr(part, "text", None)
            )
            if isinstance(text, str) and text:
                text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts)

    return None


def _prompt_type(prompt_block: Any) -> str | None:
    if isinstance(prompt_block, dict):
        prompt_type = prompt_block.get("type")
        return str(prompt_type) if isinstance(prompt_type, str) else None

    prompt_type = getattr(prompt_block, "type", None)
    return str(prompt_type) if isinstance(prompt_type, str) else None


def _prompt_content(prompt_block: Any) -> Any:
    if isinstance(prompt_block, dict):
        return prompt_block.get("content")
    return getattr(prompt_block, "content", None)


def _prompt_messages(prompt_block: Any) -> list[Any]:
    if isinstance(prompt_block, dict):
        messages = prompt_block.get("messages")
    else:
        messages = getattr(prompt_block, "messages", None)
    return messages if isinstance(messages, list) else []


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        role = message.get("role", "")
    else:
        role = getattr(message, "role", "")
    return str(role or "").lower()


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _prompt_options(param: Any) -> Any:
    if isinstance(param, dict):
        return param.get("options")
    return getattr(param, "options", None)


def _prompt_payload(param: Any) -> Any:
    if isinstance(param, dict):
        return param.get("prompt")
    return getattr(param, "prompt", None)


def extract_prompt_text_and_model(param: Any) -> tuple[str | None, str | None]:
    """Extract prompt text and embedded model from prompt-like eval parameters."""
    if param is None:
        return None, None

    prompt_text: str | None = None
    prompt_block = _prompt_payload(param)
    prompt_type = _prompt_type(prompt_block)

    if prompt_type == "completion":
        prompt_candidate = _prompt_content(prompt_block)
        if isinstance(prompt_candidate, str):
            prompt_text = prompt_candidate
    elif prompt_type == "chat":
        messages = _prompt_messages(prompt_block)
        system_messages = [
            _prompt_content_to_text(_message_content(message))
            for message in messages
            if _message_role(message) == "system"
        ]
        prompt_candidates = [text for text in system_messages if text]
        if prompt_candidates:
            prompt_text = "\n\n".join(prompt_candidates)

    options = _prompt_options(param) or {}
    model = (
        options.get("model")
        if isinstance(options, dict)
        else getattr(options, "model", None)
    )

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
        prompt_text, embedded_model = extract_prompt_text_and_model(
            params.get(prompt_key)
        )
        if prompt_text is not None:
            overrides[prompt_key] = prompt_text
        else:
            # Never forward raw prompt objects into AgentConfig string fields.
            current = overrides.get(prompt_key)
            if current is not None and not isinstance(current, str):
                overrides.pop(prompt_key, None)
        if embedded_model is not None and model_key:
            overrides[model_key] = embedded_model

    return overrides
