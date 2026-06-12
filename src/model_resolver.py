"""Helpers to resolve ADK and OpenAI clients for direct vs Gateway mode."""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from src.config import AgentConfig


def _is_gateway_enabled(config: AgentConfig | None = None) -> bool:
    if config is not None:
        return bool(config.use_gateway)
    raw = os.environ.get("BRAINTRUST_USE_GATEWAY", "false")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _gateway_url(config: AgentConfig | None = None) -> str:
    if config and config.gateway_url:
        return config.gateway_url
    return os.environ.get("BRAINTRUST_GATEWAY_URL", "https://gateway.braintrust.dev/v1")


def _gateway_api_key(config: AgentConfig | None = None) -> str | None:
    if config and config.gateway_api_key:
        return config.gateway_api_key
    return os.environ.get("BRAINTRUST_GATEWAY_API_KEY") or os.environ.get(
        "BRAINTRUST_API_KEY"
    )


def _gateway_logging_headers() -> dict[str, str]:
    """Build explicit Braintrust attribution headers for gateway requests."""
    project_id = os.environ.get("BRAINTRUST_PROJECT_ID", "").strip()
    project_name = os.environ.get("BRAINTRUST_PROJECT", "").strip()

    headers: dict[str, str] = {}
    if project_id:
        headers["x-bt-parent"] = f"project_id:{project_id}"
        headers["x-bt-project-id"] = project_id
    elif project_name:
        headers["x-bt-parent"] = f"project_name:{project_name}"
        headers["x-bt-project-name"] = project_name
    return headers


def resolve_adk_model(model_name: str, config: AgentConfig | None = None) -> Any:
    """Return either a raw model string or a Gateway-routed ADK model object."""
    if not _is_gateway_enabled(config):
        return model_name

    api_key = _gateway_api_key(config)
    if not api_key:
        raise RuntimeError(
            "BRAINTRUST_USE_GATEWAY is enabled, but no gateway key was found. "
            "Set BRAINTRUST_GATEWAY_API_KEY or BRAINTRUST_API_KEY."
        )

    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(
        model=f"openai/{model_name}",
        api_key=api_key,
        api_base=_gateway_url(config),
        extra_headers=_gateway_logging_headers(),
    )


def make_wrapped_openai_client() -> OpenAI:
    """Build OpenAI client routed through Braintrust Gateway when enabled."""
    if not _is_gateway_enabled():
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    gateway_key = _gateway_api_key()
    if not gateway_key:
        raise RuntimeError(
            "BRAINTRUST_USE_GATEWAY is enabled, but no gateway key was found. "
            "Set BRAINTRUST_GATEWAY_API_KEY or BRAINTRUST_API_KEY."
        )

    return OpenAI(
        api_key=gateway_key,
        base_url=_gateway_url(),
        default_headers=_gateway_logging_headers(),
    )
