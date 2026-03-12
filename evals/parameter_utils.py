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
        return getattr(param, "value")

    return param


def unwrap_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize all eval parameters into primitive values."""
    unwrapped: dict[str, Any] = {}
    for key, param in params.items():
        value = unwrap_parameter_value(param, default=None)
        if value is not None:
            unwrapped[key] = value
    return unwrapped
