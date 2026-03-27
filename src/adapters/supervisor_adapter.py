"""Runtime adapter that wires multi-turn chat into the ADK supervisor flow."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from src.agents.deep_agent import get_supervisor, run_supervisor_with_critic
from src.tracing import configure_adk_tracing
from src.umbrella_capabilities.multi_turn.session_store import SessionState, SessionStore

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
    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        _ = (exc_type, exc, tb)
        return False

    def log(self, *args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)

    def end(self) -> None:
        return None

    def start_span(self, **_: Any) -> "_NoopSpan":
        return _NoopSpan()


@contextmanager
def _top_level_span(name: str, input_payload: dict[str, Any], metadata: dict[str, Any]) -> Iterator[_NoopSpan]:
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


def _session_ttl_seconds() -> int:
    raw = str(os.environ.get("CHAT_SESSION_TTL_SECONDS", "1800")).strip()
    try:
        value = int(raw)
    except ValueError:
        return 1800
    return value if value > 0 else 1800


def _coerce_history_window(raw_window: Any) -> int:
    if not isinstance(raw_window, int):
        return 8
    if raw_window < 0:
        return 8
    # Keep bounded so a single request cannot explode token usage.
    return min(raw_window, 20)


def _normalize_chat_history(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _summarize_older_history(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return ""
    user_turns = sum(1 for message in messages if str(message.get("role", "")).strip().lower() == "user")
    assistant_turns = sum(1 for message in messages if str(message.get("role", "")).strip().lower() == "assistant")
    return (
        f"{user_turns} earlier user turns and {assistant_turns} assistant turns were omitted "
        "from chat_history due to history_window."
    )


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
        self._session_ttl_seconds = _session_ttl_seconds()

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

    def _reap_expired_sessions(self, now_ts: float) -> None:
        expired = self._session_store.reap_expired(now_ts=now_ts, ttl_seconds=self._session_ttl_seconds)
        for session_id, state in expired:
            self._close_session_root_span(session_id=session_id, state=state, reason="ttl")

    def _create_session_root_span(self, *, session_id: str, metadata: dict[str, Any]) -> Any | None:
        if _BT_START_SPAN is None or _BT_SPAN_TYPE is None:
            return None

        try:
            return _BT_START_SPAN(
                name=f"Chat Session: {session_id}",
                type=_BT_SPAN_TYPE.TASK,
                input={"session_id": session_id, "event": "session.start"},
                metadata={
                    "session_id": session_id,
                    "repo_id": "google-adk-supervisor",
                    "source": "chat_api",
                    "initial_metadata": dict(metadata or {}),
                },
            )
        except Exception:
            return None

    def _close_session_root_span(self, *, session_id: str, state: SessionState, reason: str) -> None:
        span = state.session_root_span
        state.session_root_span = None
        if span is None:
            return

        try:
            span.log(
                output={
                    "session_id": session_id,
                    "status": "closed",
                    "ended_by": reason,
                    "turn_count": state.turn_count,
                    "history_length": len(state.messages),
                }
            )
            span.end()
        except Exception:
            # Tracing must never break demo chat behavior.
            pass

    def _turn_span_context(
        self,
        *,
        state: SessionState,
        turn_index: int,
        session_id: str,
        message: str,
        metadata: dict[str, Any],
    ):
        span_metadata = {
            "session_id": session_id,
            "repo_id": "google-adk-supervisor",
            "turn_index": turn_index,
            "metadata": metadata,
        }
        if state.session_root_span is not None and hasattr(state.session_root_span, "start_span"):
            return state.session_root_span.start_span(
                name=f"Turn {turn_index}",
                type=_BT_SPAN_TYPE.TASK if _BT_SPAN_TYPE is not None else None,
                input={"session_id": session_id, "turn_index": turn_index, "message": message},
                metadata=span_metadata,
            )

        return _top_level_span(
            name=f"Turn {turn_index}",
            input_payload={"session_id": session_id, "turn_index": turn_index, "message": message},
            metadata=span_metadata,
        )

    def reset_session(self, session_id: str) -> bool:
        state = self._session_store.pop(session_id)
        if state is None:
            return False
        self._close_session_root_span(session_id=session_id, state=state, reason="reset")
        return True

    def _build_turn_payload(self, messages: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
        history_window = _coerce_history_window(metadata.get("history_window", 8))
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

        prior_messages = messages[:-1]
        history_slice = prior_messages[-(history_window * 2) :] if history_window else []
        chat_history = _normalize_chat_history(history_slice)

        summary_from_metadata = metadata.get("history_summary")
        history_summary = str(summary_from_metadata).strip() if isinstance(summary_from_metadata, str) else ""
        if not history_summary:
            omitted_count = max(0, len(prior_messages) - len(history_slice))
            if omitted_count:
                history_summary = _summarize_older_history(prior_messages[:omitted_count])

        return {
            "input": latest_message,
            "chat_history": chat_history,
            "history_summary": history_summary,
        }

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

        now_ts = time.time()
        self._reap_expired_sessions(now_ts)

        metadata = dict(metadata or {})
        active_session_id, state = self._session_store.resolve(session_id, now_ts=now_ts)

        if state.session_root_span is None:
            state.session_root_span = self._create_session_root_span(
                session_id=active_session_id,
                metadata=metadata,
            )

        state.turn_count += 1
        turn_index = state.turn_count
        state.messages.append({"role": "user", "content": message})

        matching_subagent = self._matching_subagent(message)

        with self._turn_span_context(
            state=state,
            turn_index=turn_index,
            session_id=active_session_id,
            message=message,
            metadata=metadata,
        ) as turn_span:
            try:
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
                    turn_payload = self._build_turn_payload(state.messages, metadata)
                    app_name = str(
                        metadata.get("workflow_name")
                        or metadata.get("app_name")
                        or f"{self._default_app_name}-{active_session_id}"
                    ).strip()
                    if not app_name:
                        app_name = f"{self._default_app_name}-{active_session_id}"

                    run_result = await run_supervisor_with_critic(
                        supervisor=self._supervisor,
                        chat_payload=turn_payload,
                        app_name=app_name,
                    )

                    assistant_message = str(run_result.get("final_output", "")).strip() or "(No response generated)"
                    critic_decision = run_result.get("critic_decision", {})
                    events = [
                        {
                            "type": "turn.completed",
                            "repo_id": "google-adk-supervisor",
                            "session_id": active_session_id,
                            "turn_index": turn_index,
                            "critic_corrected": bool(run_result.get("critic_corrected", False)),
                        }
                    ]
                    if critic_decision:
                        events.append({"type": "critic.decision", "decision": critic_decision})

                state.messages.append({"role": "assistant", "content": assistant_message})
                state.last_seen_at = time.time()
                turn_span.log(
                    output={
                        "session_id": active_session_id,
                        "assistant_message": assistant_message,
                        "events": events,
                        "history_length": len(state.messages),
                        "turn_index": turn_index,
                    }
                )
            except Exception as exc:
                turn_span.log(
                    output={
                        "session_id": active_session_id,
                        "turn_index": turn_index,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                raise

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
