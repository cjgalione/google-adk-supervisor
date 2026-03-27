import asyncio

from src.adapters import supervisor_adapter as adapter_mod
from src.api.chat_api import ChatAPI


async def _assert_raises_value_error(coro):
    try:
        await coro
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_handle_turn_reuses_session_and_includes_recent_history(monkeypatch):
    calls: list[dict] = []

    async def fake_run_supervisor_with_critic(*, supervisor, query, app_name):
        calls.append(
            {
                "supervisor": supervisor,
                "query": query,
                "app_name": app_name,
            }
        )
        return {
            "final_output": f"reply-{len(calls)}",
            "messages": [],
            "critic_decision": {"compliant": True, "required_action": "accept"},
            "critic_corrected": False,
        }

    monkeypatch.setattr(adapter_mod, "run_supervisor_with_critic", fake_run_supervisor_with_critic)
    monkeypatch.setattr(adapter_mod, "configure_adk_tracing", lambda **_: None)

    supervisor = object()
    adapter = adapter_mod.RuntimeSupervisorAdapter(supervisor=supervisor)

    first = asyncio.run(adapter.handle_turn(None, "hello", {}))
    second = asyncio.run(adapter.handle_turn(first.session_id, "what about before?", {}))

    assert first.session_id == second.session_id
    assert len(calls) == 2
    assert calls[0]["supervisor"] is supervisor
    assert calls[1]["query"].startswith("You are continuing an ongoing conversation.")
    assert "User: hello" in calls[1]["query"]
    assert "Assistant: reply-1" in calls[1]["query"]
    assert first.session_id in calls[1]["app_name"]


def test_handle_turn_routes_braintrust_queries_to_subagent(monkeypatch):
    async def should_not_run_supervisor(**_):
        raise AssertionError("Supervisor should not be called for braintrust subagent handoff")

    monkeypatch.setattr(adapter_mod, "run_supervisor_with_critic", should_not_run_supervisor)
    monkeypatch.setattr(adapter_mod, "configure_adk_tracing", lambda **_: None)

    adapter = adapter_mod.RuntimeSupervisorAdapter(supervisor=object())
    result = asyncio.run(adapter.handle_turn(None, "help with braintrust traces", {}))

    assert "Braintrust helper received" in result.assistant_message
    assert any(event.get("type") == "subagent.response" for event in result.events)


def test_chat_api_validates_payload_and_merges_workflow_name():
    class FakeAdapter:
        def __init__(self):
            self.calls = []

        async def handle_turn(self, session_id, user_input, metadata):
            self.calls.append(
                {
                    "session_id": session_id,
                    "user_input": user_input,
                    "metadata": metadata,
                }
            )
            return adapter_mod.TurnResult(
                session_id="session-1",
                assistant_message="ok",
                events=[{"type": "turn.completed"}],
            )

        def reset_session(self, session_id: str) -> bool:
            return session_id == "session-1"

    fake_adapter = FakeAdapter()
    api = ChatAPI(adapter=fake_adapter)

    asyncio.run(
        _assert_raises_value_error(
            api.chat_turn(
                {
                    "message": "   ",
                }
            )
        )
    )

    result = asyncio.run(
        api.chat_turn(
            {
                "session_id": "existing-session",
                "message": "hello",
                "workflow_name": "chat-test-flow",
                "context": {"source": "unit-test"},
            }
        )
    )

    assert result["session_id"] == "session-1"
    assert fake_adapter.calls[0]["metadata"]["source"] == "unit-test"
    assert fake_adapter.calls[0]["metadata"]["workflow_name"] == "chat-test-flow"

    reset = api.chat_reset({"session_id": "session-1"})
    assert reset == {"ok": True, "session_id": "session-1"}


def test_adapter_emits_trace_span(monkeypatch):
    captured = {"start": None, "logged": None}

    class _Span:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def log(self, *args, **kwargs):
            _ = args
            captured["logged"] = kwargs

    class _SpanType:
        TASK = "task"

    def fake_start_span(*, name, type, input, metadata):
        captured["start"] = {
            "name": name,
            "type": type,
            "input": input,
            "metadata": metadata,
        }
        return _Span()

    async def fake_run_supervisor_with_critic(*, supervisor, query, app_name):
        _ = (supervisor, query, app_name)
        return {
            "final_output": "traced output",
            "messages": [],
            "critic_decision": {},
            "critic_corrected": False,
        }

    monkeypatch.setattr(adapter_mod, "_BT_START_SPAN", fake_start_span)
    monkeypatch.setattr(adapter_mod, "_BT_SPAN_TYPE", _SpanType)
    monkeypatch.setattr(adapter_mod, "run_supervisor_with_critic", fake_run_supervisor_with_critic)
    monkeypatch.setattr(adapter_mod, "configure_adk_tracing", lambda **_: None)

    adapter = adapter_mod.RuntimeSupervisorAdapter(supervisor=object())
    result = asyncio.run(adapter.handle_turn(None, "regular question", {"source": "trace-test"}))

    assert result.assistant_message == "traced output"
    assert captured["start"] is not None
    assert captured["start"]["name"] == "chat_turn [google-adk-supervisor]"
    assert captured["logged"] is not None
    assert "output" in captured["logged"]
