"""Tracing profile controls for ADK + Braintrust instrumentation."""

from __future__ import annotations

import os

DEFAULT_TRACE_PROFILE = "full"


def get_trace_profile() -> str:
    profile = os.environ.get("TRACE_PROFILE", DEFAULT_TRACE_PROFILE).strip().lower()
    return profile if profile in {"full", "lean"} else DEFAULT_TRACE_PROFILE


def use_adk_auto_instrumentation() -> bool:
    return get_trace_profile() != "lean"


def configure_adk_tracing(
    *,
    api_key: str | None,
    project_id: str | None,
    project_name: str | None,
) -> None:
    """Configure tracing based on profile.

    - full: keep braintrust_adk setup (current behavior)
    - lean: initialize logger only and rely on explicit spans in app code
    """
    if not api_key:
        return

    if use_adk_auto_instrumentation():
        from braintrust_adk import setup_adk

        setup_adk(
            api_key=api_key,
            project_id=project_id,
            project_name=project_name,
        )
        return

    from braintrust.logger import init_logger

    init_logger(
        api_key=api_key,
        project=project_name,
        project_id=project_id,
    )
