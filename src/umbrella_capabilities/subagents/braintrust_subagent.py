"""Umbrella capability scaffold for a Braintrust-specific sub-agent."""

from __future__ import annotations

from typing import Any

from src.adapters.supervisor_adapter import RuntimeSupervisorAdapter

AGENT_ID = "braintrust_help"
ROUTING_HINTS = [
    "braintrust",
    "trace",
    "eval",
    "dataset",
    "project",
]


def default_braintrust_handler(user_input: str, metadata: dict[str, Any]) -> str:
    _ = metadata
    return (
        f"[google-adk-supervisor] Braintrust helper received: {user_input}. "
        "Wire this handler into runtime-specific Braintrust tooling as needed."
    )


def register(adapter: RuntimeSupervisorAdapter) -> None:
    adapter.register_subagent(
        agent_id=AGENT_ID,
        description="Handles Braintrust-specific questions and trace/eval triage.",
        routing_hints=ROUTING_HINTS,
        handler=default_braintrust_handler,
    )
