# ADK Eval Task Template

Use this structure for Braintrust eval tasks when runtime is Google ADK.

```python
from typing import Any

from braintrust_adk import setup_adk
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


def _extract_query(input_payload: dict[str, Any]) -> str:
    if input_payload.get("query"):
        return str(input_payload["query"])
    messages = input_payload.get("messages", [])
    if messages and isinstance(messages[0], dict) and isinstance(messages[0].get("content"), str):
        return messages[0]["content"]
    raise ValueError("Could not extract query")


def _serialize_adk_event(event: Any) -> list[dict[str, Any]]:
    # Replace with framework-specific event parsing.
    # Keep return value aligned with core output contract.
    out: list[dict[str, Any]] = []

    # Example: assistant text from final response event.
    if hasattr(event, "is_final_response") and event.is_final_response():
        text = ""
        if getattr(event, "content", None) and getattr(event.content, "parts", None):
            for part in event.content.parts:
                if getattr(part, "text", None):
                    text += str(part.text)
        out.append({"role": "assistant", "content": text})

    return out


async def run_supervisor_task(input: dict, hooks: Any = None) -> dict[str, list]:
    setup_adk(project_name="your-project-name")

    query = _extract_query(input)
    messages: list[dict[str, Any]] = [{"role": "user", "content": query}]

    # Build your ADK agent and runner.
    session_service = InMemorySessionService()
    # create session, instantiate runner, etc.

    user_message = types.Content(role="user", parts=[types.Part(text=query)])
    async for event in runner.run_async(
        user_id="eval-user",
        session_id="eval-session",
        new_message=user_message,
    ):
        messages.extend(_serialize_adk_event(event))

    return {"messages": messages}
```

## Notes

1. Keep this function signature stable for Braintrust eval.
2. Always include an assistant message, even if you synthesize from final output.
3. Add metadata to `hooks.metadata` when useful for debugging/scoring.
