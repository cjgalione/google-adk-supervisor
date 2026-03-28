import asyncio

from src.adapters import supervisor_adapter as adapter_mod
from src.api.chat_api import ChatAPI
from src.umbrella_capabilities.multi_turn import session_store as store_mod


async def _assert_raises_value_error(coro):
    try:
        await coro
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_handle_turn_reuses_session_and_includes_recent_history(monkeypatch):
    calls: list[dict] = []

    async def fake_run_supervisor_with_critic(
        *, supervisor, app_name, query=None, chat_payload=None
    ):
        calls.append(
            {
                "supervisor": supervisor,
                "query": query,
                "chat_payload": chat_payload,
                "app_name": app_name,
            }
        )
        return {
            "final_output": f"reply-{len(calls)}",
            "messages": [],
            "critic_decision": {"compliant": True, "required_action": "accept"},
            "critic_corrected": False,
        }

    monkeypatch.setattr(
        adapter_mod, "run_supervisor_with_critic", fake_run_supervisor_with_critic
    )
    monkeypatch.setattr(adapter_mod, "configure_adk_tracing", lambda **_: None)

    supervisor = object()
    adapter = adapter_mod.RuntimeSupervisorAdapter(supervisor=supervisor)

    first = asyncio.run(adapter.handle_turn(None, "hello", {}))
    second = asyncio.run(
        adapter.handle_turn(first.session_id, "what about before?", {})
    )

    assert first.session_id == second.session_id
    assert len(calls) == 2
    assert calls[0]["supervisor"] is supervisor
    assert calls[0]["chat_payload"] == {
        "input": "hello",
        "chat_history": [],
        "history_summary": "",
    }
    assert calls[1]["chat_payload"]["input"] == "what about before?"
    assert calls[1]["chat_payload"]["chat_history"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "reply-1"},
    ]
    assert calls[1]["chat_payload"]["history_summary"] == ""
    assert first.session_id in calls[1]["app_name"]


def test_handle_turn_routes_braintrust_queries_to_subagent(monkeypatch):
    async def should_not_run_supervisor(**_):
        raise AssertionError(
            "Supervisor should not be called for braintrust subagent handoff"
        )

    monkeypatch.setattr(
        adapter_mod, "run_supervisor_with_critic", should_not_run_supervisor
    )
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
    assert fake_adapter.calls[0]["session_id"] == "existing-session"

    asyncio.run(
        api.chat_turn(
            {
                "session_id": 12345,
                "message": "hello again",
            }
        )
    )
    assert fake_adapter.calls[1]["session_id"] == "12345"

    reset = api.chat_reset({"session_id": "session-1"})
    assert reset == {"ok": True, "session_id": "session-1"}


def test_adapter_emits_session_root_and_turn_hierarchy(monkeypatch):
    created: list[dict] = []

    class _SpanType:
        TASK = "task"

    class _FakeSpan:
        def __init__(self, *, name, type, input, metadata, parent=None):
            self.name = name
            self.type = type
            self.input = input
            self.metadata = metadata
            self.parent = parent
            self.logs = []
            self.ended = False
            created.append(
                {
                    "name": name,
                    "parent": getattr(parent, "name", None),
                    "span": self,
                }
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            return False

        def log(self, **kwargs):
            self.logs.append(kwargs)

        def end(self):
            self.ended = True

        def start_span(self, *, name, type, input, metadata):
            return _FakeSpan(
                name=name,
                type=type,
                input=input,
                metadata=metadata,
                parent=self,
            )

    def fake_start_span(*, name, type, input, metadata):
        return _FakeSpan(
            name=name, type=type, input=input, metadata=metadata, parent=None
        )

    async def fake_run_supervisor_with_critic(
        *, supervisor, app_name, query=None, chat_payload=None
    ):
        _ = (supervisor, app_name, query, chat_payload)
        return {
            "final_output": "traced output",
            "messages": [],
            "critic_decision": {},
            "critic_corrected": False,
        }

    monkeypatch.setattr(adapter_mod, "_BT_START_SPAN", fake_start_span)
    monkeypatch.setattr(adapter_mod, "_BT_SPAN_TYPE", _SpanType)
    monkeypatch.setattr(
        adapter_mod, "run_supervisor_with_critic", fake_run_supervisor_with_critic
    )
    monkeypatch.setattr(adapter_mod, "configure_adk_tracing", lambda **_: None)

    adapter = adapter_mod.RuntimeSupervisorAdapter(supervisor=object())
    first = asyncio.run(
        adapter.handle_turn(None, "regular question", {"source": "trace-test"})
    )
    second = asyncio.run(
        adapter.handle_turn(first.session_id, "follow up", {"source": "trace-test"})
    )

    assert first.assistant_message == "traced output"
    assert second.assistant_message == "traced output"

    roots = [row for row in created if str(row["name"]).startswith("Chat Session:")]
    turns = [row for row in created if str(row["name"]).startswith("Turn ")]
    assert len(roots) == 1
    assert len(turns) == 2
    assert turns[0]["parent"] == roots[0]["name"]
    assert turns[1]["parent"] == roots[0]["name"]

    assert adapter.reset_session(first.session_id) is True
    root_span = roots[0]["span"]
    assert root_span.ended is True
    assert any(
        log.get("output", {}).get("ended_by") == "reset" for log in root_span.logs
    )


def test_ttl_reap_closes_old_session_roots(monkeypatch):
    created: list[dict] = []

    class _SpanType:
        TASK = "task"

    class _Clock:
        def __init__(self, now: float):
            self.now = now

        def __call__(self) -> float:
            return self.now

    class _FakeSpan:
        def __init__(self, *, name, type, input, metadata, parent=None):
            self.name = name
            self.type = type
            self.input = input
            self.metadata = metadata
            self.parent = parent
            self.logs = []
            self.ended = False
            created.append(
                {
                    "name": name,
                    "parent": getattr(parent, "name", None),
                    "span": self,
                }
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            return False

        def log(self, **kwargs):
            self.logs.append(kwargs)

        def end(self):
            self.ended = True

        def start_span(self, *, name, type, input, metadata):
            return _FakeSpan(
                name=name, type=type, input=input, metadata=metadata, parent=self
            )

    def fake_start_span(*, name, type, input, metadata):
        return _FakeSpan(
            name=name, type=type, input=input, metadata=metadata, parent=None
        )

    async def fake_run_supervisor_with_critic(
        *, supervisor, app_name, query=None, chat_payload=None
    ):
        _ = (supervisor, app_name, query, chat_payload)
        return {
            "final_output": "ok",
            "messages": [],
            "critic_decision": {},
            "critic_corrected": False,
        }

    clock = _Clock(100.0)
    monkeypatch.setenv("CHAT_SESSION_TTL_SECONDS", "1")
    monkeypatch.setattr(adapter_mod, "_BT_START_SPAN", fake_start_span)
    monkeypatch.setattr(adapter_mod, "_BT_SPAN_TYPE", _SpanType)
    monkeypatch.setattr(
        adapter_mod, "run_supervisor_with_critic", fake_run_supervisor_with_critic
    )
    monkeypatch.setattr(adapter_mod, "configure_adk_tracing", lambda **_: None)
    monkeypatch.setattr(adapter_mod.time, "time", clock)
    monkeypatch.setattr(store_mod.time, "time", clock)

    adapter = adapter_mod.RuntimeSupervisorAdapter(supervisor=object())
    first = asyncio.run(adapter.handle_turn(None, "first", {}))

    clock.now = 102.0
    _ = asyncio.run(adapter.handle_turn(None, "second", {}))

    old_root_name = f"Chat Session: {first.session_id}"
    old_root_rows = [row for row in created if row["name"] == old_root_name]
    assert len(old_root_rows) == 1
    old_root_span = old_root_rows[0]["span"]
    assert old_root_span.ended is True
    assert any(
        log.get("output", {}).get("ended_by") == "ttl" for log in old_root_span.logs
    )
