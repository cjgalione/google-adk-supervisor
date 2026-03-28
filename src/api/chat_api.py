"""Framework-agnostic API surface for shared chat UI in google-adk-supervisor."""

from __future__ import annotations

from typing import Any

from src.adapters.supervisor_adapter import RuntimeSupervisorAdapter


class ChatAPI:
    def __init__(self, adapter: RuntimeSupervisorAdapter | None = None) -> None:
        self._adapter = adapter or RuntimeSupervisorAdapter()

    async def chat_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        message = str(payload.get("message", "")).strip()
        if not message:
            raise ValueError("Missing non-empty `message`")

        raw_session_id = payload.get("session_id")
        session_id = (
            str(raw_session_id).strip() if raw_session_id is not None else ""
        ) or None

        context = payload.get("context")
        metadata = dict(context) if isinstance(context, dict) else {}
        workflow_name = str(payload.get("workflow_name", "")).strip()
        if workflow_name:
            metadata.setdefault("workflow_name", workflow_name)

        result = await self._adapter.handle_turn(
            session_id=session_id,
            user_input=message,
            metadata=metadata,
        )
        return {
            "session_id": result.session_id,
            "assistant_message": result.assistant_message,
            "events": result.events,
        }

    def chat_reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            raise ValueError("session_id is required")
        ok = self._adapter.reset_session(session_id)
        return {"ok": ok, "session_id": session_id}


_default_chat_api = ChatAPI()


async def chat_turn(payload: dict[str, Any]) -> dict[str, Any]:
    return await _default_chat_api.chat_turn(payload)


def chat_reset(payload: dict[str, Any]) -> dict[str, Any]:
    return _default_chat_api.chat_reset(payload)
