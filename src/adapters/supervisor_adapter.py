"""Runtime adapter that wires multi-turn chat into the ADK supervisor flow."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from src.agents.deep_agent import get_supervisor, run_supervisor_with_critic
from src.tracing import configure_adk_tracing
from src.umbrella_capabilities.multi_turn.session_store import SessionStore

try:
    from braintrust import SpanTypeAttribute as _BT_SPAN_TYPE
    from braintrust import start_span as _BT_START_SPAN
except Exception:  # pragma: no cover - fallback when Braintrust isn't available locally
    _BT_SPAN_TYPE = None
    _BT_START_SPAN = None


@dataclass
class TurnResult:
    session_id: str
    assistant_message: str
    events: list[dict[str, Any]] = field(default_factory=list)


SubagentHandler = Callable[[str, dict[str, Any]], str]


class _NoopSpan:
    def log(self, *args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)


@contextmanager
def _start_trace_span(name: str, input_payload: dict[str, Any], metadata: dict[str, Any]) -> Iterator[_NoopSpan]:
    if _BT_START_SPAN is None or _BT_SPAN_TYPE is None:
        yield _NoopSpan()
        return

    with _BT_START_SPAN(
        name=name,
        type=_BT_SPAN_TYPE.TASK,
        input=input_payload,
        metadata=metadata,
    ) as span:
        yield span


def _format_history_messages(messages: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for message in messages:
        role = str(message.get("role", "unknown")).strip().lower() or "unknown"
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "assistant":
            label = "Assistant"
        elif role == "user":
            label = "User"
        else:
            label = role.capitalize()
        rows.append(f"{label}: {content}")
    return "\n".join(rows)


class RuntimeSupervisorAdapter:
    """Thin adapter seam backed by the existing ADK supervisor runtime."""

    def __init__(
        self,
        *,
        supervisor: Any | None = None,
        default_app_name: str = "google-adk-supervisor-chat",
    ) -> None:
        self._session_store = SessionStore()
        self._subagents: dict[str, tuple[list[str], SubagentHandler]] = {}
        self._supervisor = supervisor or get_supervisor()
        self._default_app_name = default_app_name

        self._configure_tracing()
        self._register_default_subagents()

    def _configure_tracing(self) -> None:
        if not os.environ.get("BRAINTRUST_API_KEY"):
            return
        configure_adk_tracing(
            api_key=os.environ.get("BRAINTRUST_API_KEY"),
            project_id=os.environ.get("BRAINTRUST_PROJECT_ID"),
            project_name=os.environ.get("BRAINTRUST_PROJECT", "google-adk-supervisor"),
        )

    def _register_default_subagents(self) -> None:
        try:
            from src.umbrella_capabilities.subagents.braintrust_subagent import register

            register(self)
        except Exception:
            # Keep adapter usable even if optional sub-agent registration fails.
            pass

    def reset_session(self, session_id: str) -> bool:
        return self._session_store.reset(session_id)

    def _build_contextual_query(self, messages: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
        history_window = metadata.get("history_window", 8)
        if not isinstance(history_window, int) or history_window < 0:
            history_window = 8

        latest_message = next(
            (
                str(message.get("content", "")).strip()
                for message in reversed(messages)
                if str(message.get("role", "")).strip().lower() == "user"
            ),
            "",
        )
        if not latest_message:
            raise ValueError("No user message available for the current turn")

        # Use recent turns to provide multi-turn continuity while avoiding giant prompts.
        prior_messages = messages[:-1]
        history_slice = prior_messages[-(history_window * 2) :] if history_window else []
        history_text = _format_history_messages(history_slice)

        if not history_text:
            return latest_message

        return (
            "You are continuing an ongoing conversation.\n"
            "Use the previous turns as context, then answer the latest user message directly.\n\n"
            f"Conversation so far:\n{history_text}\n\n"
            f"Latest user message:\n{latest_message}"
        )

    def _matching_subagent(self, user_input: str) -> tuple[str, SubagentHandler] | None:
        lowered = user_input.lower()
        for agent_id, (routing_hints, handler) in self._subagents.items():
            if any(hint.lower() in lowered for hint in routing_hints):
                return agent_id, handler
        return None

    async def handle_turn(
        self,
        session_id: str | None,
        user_input: str,
        metadata: dict[str, Any],
    ) -> TurnResult:
        message = str(user_input or "").strip()
        if not message:
            raise ValueError("user_input must be non-empty")

        active_session_id, state = self._session_store.resolve(session_id)
        state.messages.append({"role": "user", "content": message})

        metadata = dict(metadata or {})
        matching_subagent = self._matching_subagent(message)

        with _start_trace_span(
            name="chat_turn [google-adk-supervisor]",
            input_payload={"session_id": active_session_id, "message": message},
            metadata={
                "session_id": active_session_id,
                "repo_id": "google-adk-supervisor",
                "turn_index": len(state.messages),
                "metadata": metadata,
            },
        ) as turn_span:
            if matching_subagent is not None:
                agent_id, handler = matching_subagent
                assistant_message = str(handler(message, metadata)).strip() or "(No response generated)"
                events: list[dict[str, Any]] = [
                    {
                        "type": "subagent.response",
                        "agent_id": agent_id,
                        "session_id": active_session_id,
                    }
                ]
            else:
                contextual_query = self._build_contextual_query(state.messages, metadata)
                app_name = str(
                    metadata.get("workflow_name")
                    or metadata.get("app_name")
                    or f"{self._default_app_name}-{active_session_id}"
                ).strip()
                if not app_name:
                    app_name = f"{self._default_app_name}-{active_session_id}"

                run_result = await run_supervisor_with_critic(
                    supervisor=self._supervisor,
                    query=contextual_query,
                    app_name=app_name,
                )

                assistant_message = str(run_result.get("final_output", "")).strip() or "(No response generated)"
                critic_decision = run_result.get("critic_decision", {})
                events = [
                    {
                        "type": "turn.completed",
                        "repo_id": "google-adk-supervisor",
                        "session_id": active_session_id,
                        "critic_corrected": bool(run_result.get("critic_corrected", False)),
                    }
                ]
                if critic_decision:
                    events.append({"type": "critic.decision", "decision": critic_decision})

            state.messages.append({"role": "assistant", "content": assistant_message})
            turn_span.log(
                output={
                    "session_id": active_session_id,
                    "assistant_message": assistant_message,
                    "events": events,
                    "history_length": len(state.messages),
                }
            )

        return TurnResult(
            session_id=active_session_id,
            assistant_message=assistant_message,
            events=events,
        )

    def register_subagent(
        self,
        agent_id: str,
        description: str,
        routing_hints: list[str],
        handler: SubagentHandler,
    ) -> None:
        _ = description
        self._subagents[agent_id] = (list(routing_hints), handler)
