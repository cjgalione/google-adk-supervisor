"""Runtime helpers for Google ADK runs and eval serialization."""

from __future__ import annotations

import json
import uuid
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


def extract_query_from_input(input_payload: dict[str, Any]) -> str:
    """Extract a user query from eval input payloads."""
    if "query" in input_payload and input_payload["query"]:
        return str(input_payload["query"])

    messages = input_payload.get("messages", [])
    if isinstance(messages, list) and messages:
        first_message = messages[0]
        if isinstance(first_message, dict):
            content = first_message.get("content")
            if isinstance(content, str):
                return content

    raise ValueError("Could not extract user query from input payload")


def _part_text(part: Any) -> str:
    text = getattr(part, "text", None)
    if isinstance(text, str):
        return text
    return ""


def _part_function_call(part: Any) -> tuple[str, Any] | None:
    function_call = getattr(part, "function_call", None)
    if function_call is None:
        return None

    name = str(getattr(function_call, "name", "") or "")
    args = getattr(function_call, "args", {})
    return name, args


def _part_function_response(part: Any) -> tuple[str, Any] | None:
    function_response = getattr(part, "function_response", None)
    if function_response is None:
        return None

    name = str(getattr(function_response, "name", "") or "")
    response = getattr(function_response, "response", "")
    return name, response


def _safe_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None), list, dict)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()  # type: ignore[attr-defined]
    try:
        return json.loads(json.dumps(value))
    except Exception:
        return str(value)


def _serialize_event(event: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return out

    assistant_text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for part in parts:
        text = _part_text(part)
        if text:
            assistant_text_parts.append(text)

        fc = _part_function_call(part)
        if fc is not None:
            tool_name, args = fc
            if tool_name:
                tool_calls.append({"name": tool_name, "args": _safe_json(args)})

        fr = _part_function_response(part)
        if fr is not None:
            _, response = fr
            out.append(
                {
                    "role": "tool",
                    "content": response if isinstance(response, str) else str(_safe_json(response)),
                }
            )

    if tool_calls:
        out.insert(
            0,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls,
            },
        )

    if assistant_text_parts:
        out.append(
            {
                "role": "assistant",
                "content": "\n".join(assistant_text_parts).strip(),
            }
        )

    return out


async def run_adk_agent(
    *,
    agent: Any,
    query: str,
    app_name: str,
    user_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run an ADK agent and return final text plus serialized messages."""
    uid = user_id or "eval-user"
    sid = session_id or f"session-{uuid.uuid4().hex}"

    session_service = InMemorySessionService()
    await session_service.create_session(app_name=app_name, user_id=uid, session_id=sid)

    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)
    user_msg = types.Content(role="user", parts=[types.Part(text=query)])

    messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
    final_output = ""

    async for event in runner.run_async(user_id=uid, session_id=sid, new_message=user_msg):
        event_messages = _serialize_event(event)
        messages.extend(event_messages)

        if hasattr(event, "is_final_response") and event.is_final_response():
            content = getattr(event, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if parts:
                text_parts = [
                    _part_text(part).strip()
                    for part in parts
                    if _part_text(part).strip()
                ]
                if text_parts:
                    final_output = "\n".join(text_parts)

    if final_output and not any(
        m.get("role") == "assistant" and m.get("content") for m in messages
    ):
        messages.append({"role": "assistant", "content": final_output})

    return {"final_output": final_output, "messages": messages}
